import pytest

from models import TreeNode, DDTN, FDDTN
import torch.nn as nn
import torch.nn.functional as F
import torch


@pytest.fixture
def stump():
    labels = ['zero', 'one']
    left = TreeNode(2, 2, None, None, labels)
    right = TreeNode(2, 2, None, None, labels)
    stump = TreeNode(2, 2, left, right, ['1-sigma', 'sigma'])
    stump.beta = nn.Parameter(torch.Tensor([[1], [0]]))
    stump.phi = nn.Parameter(torch.Tensor([0]))
    left.beta = nn.Parameter(torch.Tensor([1, 0]))
    right.beta = nn.Parameter(torch.Tensor([0, 1]))
    return stump


@pytest.fixture
def tree_stump():
    tree_stump = DDTN(2, 2, 2, True, ['zero', 'one'])
    tree_stump.all_nodes[0].beta = nn.Parameter(torch.Tensor([1, 0]))
    tree_stump.all_nodes[1].beta = nn.Parameter(torch.Tensor([0, 1]))
    tree_stump.all_nodes[2].beta = nn.Parameter(torch.Tensor([[1], [0]]))
    tree_stump.all_nodes[2].phi = nn.Parameter(torch.Tensor([0]))
    return tree_stump


@pytest.fixture
def trees():
    fast_tree = FDDTN(3, 2, 4, True, ['zero', 'one'])
    leafs = torch.zeros_like(fast_tree.leafs)
    for i in range(leafs.shape[0]):
        leafs[i, i] = 1
    fast_tree.leafs = nn.Parameter(leafs)

    # level 1 Beta
    level1_beta = torch.zeros_like(fast_tree.nodes_beta[0])
    level1_beta[0, 1] = 1
    level1_beta[1] = 0.5
    fast_tree.nodes_beta[0] = nn.Parameter(level1_beta)

    # level 1 phi
    level1_phi = torch.zeros_like(fast_tree.nodes_phi[0])
    fast_tree.nodes_phi[0] = nn.Parameter(level1_phi)

    # level 2 Beta
    level0_beta = torch.zeros_like(fast_tree.nodes_beta[1])
    level0_beta[0, 0] = 1
    fast_tree.nodes_beta[1] = nn.Parameter(level0_beta)

    # level 2 phi
    level2_phi = torch.zeros_like(fast_tree.nodes_phi[1])
    fast_tree.nodes_phi[1] = nn.Parameter(level2_phi)

    # recursive DDT Implementation
    slow_tree = DDTN(3, 2, 4, True, ['zero', 'one'])
    for i in range(leafs.shape[0]):
        slow_tree.all_nodes[i].beta = nn.Parameter(leafs[i])
    for i in range(4, len(slow_tree.all_nodes)):
        slow_tree.all_nodes[i].phi = nn.Parameter(
            torch.zeros_like(slow_tree.all_nodes[i].phi))
    slow_tree.all_nodes[4].beta = nn.Parameter(level1_beta[0].unsqueeze(1))
    slow_tree.all_nodes[5].beta = nn.Parameter(level1_beta[1].unsqueeze(1))
    slow_tree.all_nodes[6].beta = nn.Parameter(level0_beta[0].unsqueeze(1))
    return fast_tree, slow_tree


def test_crisp_tree(trees):
    fast_tree, slow_tree = trees
    fast_tree.nodes_beta[0][1, 1] += .1
    input = torch.Tensor([
        [-100, -100],
        [-100, 100],
        [100, -1e2]
,        [100, 100]])
    expected = torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    fast_tree.make_hard()
    output = fast_tree(input)
    torch.testing.assert_allclose(output, expected)


def test_leaf_regularization(trees):
    fast_tree, slow_tree = trees
    expected_loss = -1.0
    fast_loss = fast_tree.leaf_regularization().item()
    assert expected_loss == fast_loss


def test_tree_regregularization(trees):
    fast_tree, slow_tree = trees
    expected_loss = torch.tensor(-2.0 / 3)
    fast_loss = fast_tree.regregularization()
    torch.testing.assert_allclose(fast_loss, expected_loss)


def test_tree_alpha(trees):
    # TODO: alpha should sharpen to the point where ambiguityu in decision
    # vanishes
    assert True


def test_tree_prunning(trees):
    # TODO: tree prunning should copy the pruned to lower levels.
    assert True


def test_tree_regularization(trees):
    fast_tree, slow_tree = trees
    fast_loss = fast_tree.regularization().item()
    slow_loss = slow_tree.regularization().item()
    expected_loss = -2.0
    assert expected_loss == fast_loss
    assert expected_loss == slow_loss


def test_tree_to_png(trees):
    import os
    os.makedirs('test_output', exist_ok=True)
    action_labels = ['y_0', 'y_1', 'y_2', 'y_3']
    rec_tree_file = os.path.join('test_output', 'rec_tree.png')
    fast_tree_file = os.path.join('test_output', 'fast_tree.png')
    disc_tree_file = os.path.join('test_output', 'disc_tree.png')
    disc_disc_tree_file = os.path.join('test_output', 'disc_disc_tree.png')
    fast_tree, slow_tree = trees
    slow_tree.tree_to_png(rec_tree_file)
    fast_tree.action_labels = action_labels
    fast_tree.hard_tree_to_png(disc_tree_file)
    fast_tree.tree_to_png(fast_tree_file)
    fast_tree.continuous = False
    fast_tree.hard_tree_to_png(disc_disc_tree_file)


def test_batch_continuous(trees):
    fast_tree, slow_tree = trees
    input = torch.Tensor([
        [-100, -100],
        [-100, 100],
        [100, 100],
        [100, -1e2]])
    expected = torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, .5, .5]])
    fast_tree.make_hard()
    fast_tree.make_soft()
    fast_out = fast_tree(input)
    slow_out = slow_tree(input)
    torch.testing.assert_allclose(fast_out, expected)
    torch.testing.assert_allclose(slow_out, expected)


def test_batch_discrete(trees):
    fast_tree, slow_tree = trees
    fast_tree.continuous = False
    slow_tree.continuous = False
    input = torch.Tensor([
        [-100, -100],
        [-100, 100],
        [100, 100],
        [100, -1e2]])
    expected = F.softmax(torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, .5, .5]]), dim=1)

    fast_out = fast_tree(input)
    slow_out = slow_tree(input)
    torch.testing.assert_allclose(fast_out, expected)
    torch.testing.assert_allclose(slow_out, expected)


@pytest.mark.parametrize('input,expected',
                         [
                             (torch.Tensor([-100, -100]),
                              torch.Tensor([1, 0, 0, 0])),
                             (torch.Tensor([-100, 100]),
                              torch.Tensor([0, 1, 0, 0])),
                             (torch.Tensor([100, 100]),
                              torch.Tensor([0, 0, 0, 1])),
                             (torch.Tensor([100, -1e2]),
                              torch.Tensor([0, 0, .5, .5])),
                         ])
def test_fast_tree(trees, input, expected):
    fast_tree, slow_tree = trees
    fast_out = fast_tree(input)
    slow_out = slow_tree(input)
    torch.testing.assert_allclose(fast_out, expected)
    torch.testing.assert_allclose(slow_out, expected)


@pytest.mark.parametrize('input,expected',
                         [
                             (torch.Tensor([-100, -100]),
                              F.softmax(torch.Tensor([1, 0, 0, 0]), dim=-1)),
                             (torch.Tensor([-100, 100]),
                              F.softmax(torch.Tensor([0, 1, 0, 0]), dim=-1)),
                             (torch.Tensor([100, 100]),
                              F.softmax(torch.Tensor([0, 0, 0, 1]), dim=-1)),
                             (torch.Tensor([100, -1e2]),
                              F.softmax(torch.Tensor([0, 0, .5, .5]), dim=-1)),
                         ])
def test_fast_tree_cntn(trees, input, expected):
    fast_tree, slow_tree = trees
    fast_tree.continuous = False
    slow_tree.continuous = False
    fast_out = fast_tree(input)
    slow_out = slow_tree(input)
    torch.testing.assert_allclose(fast_out, expected)
    torch.testing.assert_allclose(slow_out, expected)


@pytest.mark.parametrize('input,expected',
                         [
                             (torch.Tensor([-100, 0]), torch.Tensor([1, 0])),
                             (torch.Tensor([100, 0]), torch.Tensor([0, 1])),
                             (torch.Tensor([0, 0]), torch.Tensor([.5, .5]))
                         ])
def test_fast_stump(input, expected):
    fast_stump = FDDTN(2, 2, 2, True, ['zero', 'one'])
    fast_stump.leafs[0] = torch.Tensor([1, 0])
    fast_stump.leafs[1] = torch.Tensor([0, 1])
    fast_stump.nodes_beta[0] = nn.Parameter(torch.Tensor([[1, 0]]))
    fast_stump.nodes_phi = nn.Parameter(torch.Tensor([0]))
    actual = fast_stump(input)
    torch.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('input,expected',
                         [
                             (torch.Tensor([-100, 0]), torch.Tensor([1, 0])),
                             (torch.Tensor([100, 0]), torch.Tensor([0, 1])),
                             (torch.Tensor([0, 0]), torch.Tensor([.5, .5]))
                         ])
def test_tree_stump(tree_stump, input, expected):
    actual = tree_stump(input)
    torch.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('input,expected',
                         [
                             (torch.Tensor([-100, 0]), torch.Tensor([1, 0])),
                             (torch.Tensor([100, 0]), torch.Tensor([0, 1])),
                             (torch.Tensor([0, 0]), torch.Tensor([.5, .5]))
                         ])
def test_stump(stump, input, expected):
    actual = stump(input)
    torch.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0.0)
