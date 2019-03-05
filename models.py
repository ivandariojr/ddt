import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ete3 import Tree, NodeStyle, faces


class LinearPolicy(nn.Module):

    def __init__(self, n_input, n_output, continuous):
        super().__init__()
        self.continuous = continuous
        self.l1 = nn.Linear(n_input, n_output)
        self.alpha = 0

    def regularization(self):
        return 0

    def forward(self, x):
        if self.continuous:
            return self.l1(x)
        return F.softmax(self.l1(x), dim=1)


class MLP(nn.Module):

    def __init__(self, n_input, n_output, continuous):
        super().__init__()
        self.continuous = continuous
        self.n_input = n_input
        self.n_output = n_output
        self.l1 = nn.Linear(n_input, 256)
        self.l2 = nn.Linear(256, 512)
        self.l3 = nn.Linear(512, n_output)
        self.alpha = 0

    def regularization(self):
        return 0

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if self.continuous:
            return self.l3(x)
        return F.softmax(self.l3(x), dim=1)


class TreeNode(nn.Module):

    def __init__(self, n_input, n_outputs, child1, child2, labels):
        super().__init__()
        self.labels = labels
        self.child1 = child1
        self.child2 = child2
        if self.child1 is not None and self.child2 is not None:
            self.beta = nn.Parameter(torch.rand(n_input, 1))
            self.phi = nn.Parameter(torch.rand(1))
        else:
            self.beta = nn.Parameter(torch.rand(n_outputs))
            self.phi = None
        # self.beta = nn.Parameter(torch.FloatTensor([[0], [0], [1000], [0]]))
        # self.phi = nn.Parameter(torch.zeros(1))

    def regularization(self):
        if self.child1 is None and self.child2 is None:
            return torch.Tensor([0])
        non_max_idx = torch.ones_like(self.beta).byte()
        argmax_beta = torch.argmax(self.beta)
        non_max_idx[argmax_beta] = 0
        beta_non_max = self.beta[non_max_idx].squeeze()
        beta_max = self.beta[argmax_beta]
        return beta_non_max.norm(1) - beta_max.norm(1)
        # return beta_non_max.norm(1) + beta_non_max.norm(2) - beta_max.norm(2) - beta_max.norm(2)

    # param_vector = torch.cat([, self.phi])
    # return param_vector.norm(2)
    def node_viz(self):
        thisNode = Tree()
        thisNode.set_style(NodeStyle(shape='square'))
        if self.child2 is not None:
            thisNode.add_child(self.child2.node_viz())
        if self.child1 is not None:
            thisNode.add_child(self.child1.node_viz())
        if self.child1 is not None and self.child2 is not None:
            thisNode.add_face(
                faces.BarChartFace(
                    self.beta.squeeze().detach().cpu().numpy(),
                    labels=self.labels,
                    min_value=0.0,
                    max_value=self.beta.max().detach().cpu().item()), 0)
            thisNode.add_face(
                faces.TextFace("phi: {0:.3f}".format(self.phi.item())), 0)
        else:
            thisNode.add_face(
                faces.BarChartFace(
                    self.beta.squeeze().detach().cpu().numpy(),
                    min_value=0.0,
                    max_value=self.beta.max().detach().cpu().item()), 0)
        return thisNode

    def forward(self, x):

        if self.child1 == None and self.child2 == None:
            return self.beta

        sig = torch.sigmoid(x @ self.beta + self.phi)
        if torch.any(torch.isnan(sig)).detach().cpu().item() != 0:
            raise Exception('[ERROR] Invalid sig')
        return (1 - sig) * self.child1(x) + sig * self.child2(x)


class FDDTN(nn.Module):
    def __init__(self, depth, n_input, n_output, continuous, labels,
                 param_initer=torch.rand, init_alpha=1.0, action_labels=None):
        super().__init__()
        self.continuous = continuous
        self.n_input = n_input
        self.n_output = n_output
        self.labels = labels
        self.depth = depth
        self.leafs = nn.Parameter(param_initer(2 ** (self.depth - 1), n_output))
        self.nodes_beta = nn.ParameterList()
        self.nodes_phi = nn.ParameterList()
        self.alpha = init_alpha
        self.action_labels = action_labels
        self.hard_mode = False
        if action_labels is not None:
            # you need the same number of action labels as number of outputs
            assert len(action_labels) == n_output
            #this needs to be a regular python list for rendering
            assert isinstance(action_labels, list)
        if labels is not None:
            # you need the same number of labels as inputs
            assert len(labels) == n_input
            #this needs to be a regular python list for rendering
            assert isinstance(labels, list)
        for d in range(depth - 1, 0, -1):
            self.nodes_beta.append(
                nn.Parameter(param_initer(2 ** (d - 1), n_input)))
            self.nodes_phi.append(nn.Parameter(param_initer(2 ** (d - 1), 1)))

    def make_hard(self):
        self.hard_mode = True

    def make_soft(self):
        self.hard_mode = False

    def _separate_max(self, tensor):
        non_max_idx = torch.ones_like(tensor).byte()
        argmaxes = torch.argmax(tensor, dim=1)
        #this feels like a pytorch bug
        non_max_idx[[range(non_max_idx.shape[0])], argmaxes] = 0
        non_maxes = tensor[non_max_idx]
        maxes = tensor[[range(tensor.shape[0])], argmaxes]
        return non_maxes, maxes

    def regularization(self):
        betas = torch.cat([t for t in self.nodes_beta], dim=0)
        beta_non_max, beta_max = self._separate_max(betas)
        return beta_non_max.norm(1) - beta_max.norm(1)

    def regregularization(self):
        betas = torch.cat([t for t in self.nodes_beta], dim=0)
        beta_non_max, beta_max = self._separate_max(betas)
        return (beta_non_max.norm(1) / float(np.prod(beta_non_max.shape))) - \
               (beta_max.norm(1) / float(np.prod(beta_max.shape)))

    def leaf_regularization(self):
        leaf_non_max, leaf_max = self._separate_max(self.leafs)
        return (leaf_non_max.norm(1) / float(np.prod(leaf_non_max.shape))) - \
               (leaf_max.norm(1) / float(np.prod(leaf_max.shape)))

    def hard_tree_to_png(self, filepath: str):
        nodes = list()
        leaf_tmpl = '{}: y_{}'
        for i in range(self.leafs.shape[0]):
            thisNode = Tree()
            if self.continuous:
                thisNode.add_face(
                    faces.BarChartFace(
                        self.leafs[i].detach().cpu().numpy(),
                        min_value=0.0,
                        max_value=self.leafs[i].max().detach().cpu().numpy() + 1e-7,
                        labels=self.action_labels
                    ), 0)
            else:
                max_leaf_idx = np.argmax(self.leafs[i].detach().cpu().numpy())
                thisNode.add_face(faces.TextFace(
                    leaf_tmpl.format(
                        self.action_labels[max_leaf_idx],
                        max_leaf_idx)), 0)
            nodes.append(thisNode)
        node_tmpl = '{}: x_{} >= {}'
        for d in range(self.depth-1):
            for node_i in range(self.nodes_beta[d].shape[0]):
                thisNode = Tree()
                thisNode.add_child(nodes.pop(1))
                thisNode.add_child(nodes.pop(0))
                beta = F.softmax(self.nodes_beta[d][node_i].squeeze(), 0
                                 ).detach().cpu().numpy()
                phi = self.nodes_phi[d][node_i].squeeze().detach().cpu().item()
                max_beta_idx = np.argmax(beta)


                thisNode.add_face(faces.TextFace(node_tmpl.format(
                    self.labels[max_beta_idx],
                    max_beta_idx,
                    phi)), 0)
                nodes.append(thisNode)
        if filepath is not None:
            nodes[0].render(filepath,)
        return nodes[0]

    def tree_to_png(self, filepath: str):
        nodes = list()
        for i in range(self.leafs.shape[0]):
            thisNode = Tree()
            thisNode.add_face(
                faces.BarChartFace(
                    self.leafs[i].detach().cpu().numpy(),
                    labels=self.action_labels,
                    min_value=0.0,
                    max_value=self.leafs[i].max().detach().cpu().numpy() + 1e-7,
                ), 0)
            nodes.append(thisNode)
        for d in range(self.depth-1):
            for node_i in range(self.nodes_beta[d].shape[0]):
                thisNode = Tree()
                thisNode.add_child(nodes.pop(1))
                thisNode.add_child(nodes.pop(0))
                beta = F.softmax(self.nodes_beta[d][node_i].squeeze(), 0
                                 ).detach().cpu().numpy()
                phi = self.nodes_phi[d][node_i].squeeze().detach().cpu().item()
                thisNode.add_face(
                    faces.BarChartFace(
                        beta,
                        labels=self.labels,
                        min_value=0.0,
                        max_value=1.0
                    ), 0)
                thisNode.add_face(
                    faces.TextFace('phi: {0:.3f}'.format(phi)), 0)
                nodes.append(thisNode)
        if filepath is not None:
            nodes[0].render(filepath)
        return nodes[0]

    def print_tree_weights(self, writer):
        pass

    def hard_forward(self, x):
        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        n_batch = x.shape[0]
        iterate = self.leafs.unsqueeze(0).expand(n_batch, -1, -1)
        for betas, phis in zip(self.nodes_beta, self.nodes_phi):
            max_decision = torch.argmax(betas, 1)
            chosen_vars = x[:, max_decision]
            sig = chosen_vars >= phis.squeeze().unsqueeze(0).expand_as(chosen_vars)
            sig = sig.type_as(iterate).unsqueeze(2)

            lefts = (1 - sig).expand_as(iterate[:, 0::2]) * \
                    iterate[:, 0::2]
            rights = sig.expand_as(iterate[:, 1::2]) * \
                     iterate[:, 1::2]
            iterate = lefts + rights
        iterate = iterate.squeeze()
        if not self.continuous:
            return F.softmax(iterate, dim=-1)
        return iterate

    def soft_forward(self, x):
        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        n_batch = x.shape[0]
        iterate = self.leafs.unsqueeze(0).expand(n_batch, -1, -1)
        for betas, phis in zip(self.nodes_beta, self.nodes_phi):
            beta_mat = F.softmax(betas, 1).unsqueeze(0).expand(n_batch, -1, -1)
            x_mat = x.unsqueeze(2)
            sig = torch.sigmoid( self.alpha * (
                torch.matmul(beta_mat, x_mat) + phis.unsqueeze(0)))
            lefts = (1 - sig).expand_as(iterate[:, 0::2]) * \
                    iterate[:, 0::2]
            rights = sig.expand_as(iterate[:, 1::2]) * \
                     iterate[:, 1::2]
            iterate = lefts + rights
        iterate = iterate.squeeze()
        if not self.continuous:
            return F.softmax(iterate, dim=-1)
        return iterate

    def forward(self, x):
        if self.hard_mode:
            return self.hard_forward(x)
        else:
            return self.soft_forward(x)


class DDTN(nn.Module):
    def __init__(self, depth, n_input, n_output, continuous, labels):
        super().__init__()
        self.all_nodes = nn.ModuleList()
        self.continuous = continuous
        self.n_input = n_input
        self.n_output = n_output
        for d in range(depth):
            new_level = []
            for node in range(2 ** (depth - d - 1)):
                if d != 0:
                    c1, c2 = (prev_children.pop(0), prev_children.pop(0))
                else:
                    c1, c2 = (None, None)
                added_tree = TreeNode(n_input, n_output, c1, c2, labels)
                new_level.append(added_tree)
                self.all_nodes.append(added_tree)
            prev_children = new_level
        self.tree = prev_children[0]

    def regularization(self):
        loss = 0.0
        for n in self.all_nodes:
            loss = loss + n.regularization()
        return loss

    def tree_to_png(self, filepath: str):
        tree_viz = self.tree.node_viz()
        if filepath is not None:
            tree_viz.render(filepath)
        return tree_viz

    def print_tree_weights(self, writer):
        for n_i in range(len(self.all_nodes)):
            writer.add_histogram('%d_beta' % n_i, self.all_nodes[n_i].beta, )
            writer.add_histogram('%d_phi' % n_i, self.all_nodes[n_i].phi)

    def forward(self, x):
        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        tree_out = self.tree(x)
        if self.continuous:
            return tree_out
        return F.softmax(tree_out, dim=1)

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.output_size = actor.n_output

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return 1

    def _forward_gru(self, x, hxs, masks):
        raise Exception('[ERROR] Why is this being called?')

    def forward(self, inputs, rnn_hxs, masks):
        return self.critic(inputs), self.actor(inputs), rnn_hxs
