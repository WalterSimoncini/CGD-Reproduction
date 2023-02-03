import torch
import math
import tqdm
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

class CG(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train, group_info):
        # initialize model
        model = initialize_model(config, d_out).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.logged_fields.append('group_alpha')
        
        # step size
        # self.rho = config.metacg_rho
        self.outer_lr = config.lr
        
        self.C = config.cg_C
        _, self.g_counts = group_info
        
        self.g_counts = torch.tensor(self.g_counts, device=self.device, dtype=torch.float32)
        wts = torch.exp(self.C/torch.sqrt(self.g_counts))
        self.wts = wts/wts.sum()
        self.adj_wts = self.C/torch.sqrt(self.g_counts)
        print ("Using up-weight: ", self.wts.cpu().numpy())
        print ("Using adj-weight: ", self.adj_wts.cpu().numpy())
        
        self.config = config
        self.batch_size = config.batch_size
        self.device = config.device
        self.num_groups = grouper.n_groups
        self.num_train_groups = is_group_in_train.sum().item()
        self.is_group_in_train = is_group_in_train

        self.alpha = torch.autograd.Variable(torch.ones(self.num_groups, device=self.device)*(1./self.num_groups), requires_grad=True)
        
        self.lmbda = torch.autograd.Variable(torch.zeros(self.num_groups, device=self.device), requires_grad=True)
        
        self.rwt = torch.autograd.Variable(self.wts.clone(), requires_grad=False)
        self.step_size = config.cg_step_size
        self.rloss = torch.zeros_like(self.g_counts)

        # 5 groups --> alpha = [0.2, 0.2, 0.2, 0.2, 0.2]

    def process_batch(self, batch, unlabeled_batch=None):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
              all Tensors are of size (batch_size,)
        """
        results = super().process_batch(batch)
        results['group_alpha'] = self.alpha.detach()
        return results

    def objective(self, results):
        # compute group losses
        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.num_groups,
            return_dict=False)

        group_losses = group_losses[self.is_group_in_train]
        loss = group_losses @ self.rwt

        return loss

    def _params(self):
        if self.config.model.find('bert') >= 0:
            params = []
            select = ['layer.10', 'layer.11', 'bert.pooler.dense', 'classifier']
            for name, param in self.model.named_parameters():
                for s in select:
                    if (name.find(s) >= 0):
                        params.append(param)
                        break
            return params
        elif self.config.model.startswith('resnet50'):
            params = []
            select = ['layer3', 'layer4', 'fc.weight', 'fc.bias']
            for name, param in self.model.named_parameters():
                for s in select:
                    if (name.find(s) >= 0):
                        params.append(param)
                        break
            return params
        elif self.config.model.startswith('densenet121'):
            params = []
            select = ['features.denseblock4', 'classifier']
            for name, param in self.model.named_parameters():
                for s in select:
                    if (name.find(s) >= 0):
                        params.append(param)
                        break
            return params
        else:
            return list(self.model.parameters())  
    
    def _update(self, results, should_step=True):
        """
        Process the batch, update the log, and update the model, group weights, and scheduler.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
                - objective (float)
        """      
        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.num_groups,
            return_dict=False)        
        params = self._params()
        all_grads = [None]*self.num_groups
        for li in range(self.num_groups):
            all_grads[li] = torch.autograd.grad(group_losses[li], params, retain_graph=True)
            assert all_grads[li] is not None
        
        RTG = torch.zeros([self.num_groups, self.num_groups], device=self.device)
        for li in range(self.num_groups):
            for lj in range(self.num_groups):
                dp = 0
                vec1_sqnorm, vec2_sqnorm = 0, 0
                for pi in range(len(params)):
                    fvec1 = all_grads[lj][pi].detach().flatten()
                    fvec2 = all_grads[li][pi].detach().flatten()
                    dp += fvec1 @ fvec2
                    vec1_sqnorm += torch.norm(fvec1)**2
                    vec2_sqnorm += torch.norm(fvec2)**2
                RTG[li, lj] = dp/torch.clamp(torch.sqrt(vec1_sqnorm*vec2_sqnorm), min=1e-3)

        _gl = torch.sqrt(group_losses.detach().unsqueeze(-1))

        RTG = torch.mm(_gl, _gl.t()) * RTG
        # Select only relevant (train) groups
        RTG = RTG[:, self.is_group_in_train][self.is_group_in_train, :]

        _exp = self.step_size*(RTG @ self.wts)
        
        # to avoid overflow
        _exp -= _exp.max()

        train_groups = self.is_group_in_train.nonzero(as_tuple=True)[0]

        self.alpha.data[train_groups] = torch.exp(_exp)
        self.rwt *= self.alpha.data[train_groups]
        self.rwt = self.rwt/self.rwt.sum()
        self.rwt = torch.clamp(self.rwt, min=1e-5)

        # The logger requires an alpha value for each group,
        # regardless of whether they're not used for training
        group_alpha = torch.zeros(self.num_groups, device=self.device)
        group_alpha[train_groups] = self.rwt.detach()

        results['group_alpha'] = group_alpha
        # update model
        super()._update(results, should_step=should_step)
