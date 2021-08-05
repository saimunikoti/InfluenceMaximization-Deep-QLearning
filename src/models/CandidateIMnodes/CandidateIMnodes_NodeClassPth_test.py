import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from src.data import config as cnf
import torch.optim as optim
import time
import argparse

from src.models.CandidateIMnodes.GNNmodel import SAGE
from src.data.load_graph import load_plcgraph, inductive_split
from src.data import utils as ut
from src.data import config
import pickle
import warnings
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
warnings.filterwarnings("ignore")
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

##
def evaluate_test(model, test_nfeat, test_labels, device, dataloader, loss_fcn):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval() # change the mode

    test_acc = 0.0
    test_loss = 0.0
    class1acc = 0.0
    y_test = []
    y_pred = []

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

        with th.no_grad():

            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            temp_pred = th.argmax(batch_pred, dim=1)

            y_test.append(batch_labels.cpu().detach().numpy())
            y_pred.append(temp_pred.cpu().detach().numpy())

            current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy() )
            test_acc = test_acc + ((1 / (step + 1)) * (current_acc - test_acc))

            cnfmatrix = confusion_matrix(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
            class1acc = class1acc + ((1 / (step + 1)) * (cnfmatrix[0][0] / np.sum(cnfmatrix[0, :]) - class1acc))

            print(cnfmatrix)

            # correct = temp_pred.eq(batch_labels)
            # test_acc = test_acc + correct

            loss = loss_fcn(batch_pred, batch_labels)
            test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

    model.train() # rechange the model mode to training

    return test_acc, test_loss, class1acc, y_test, y_pred

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    th.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = th.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, valid_loss_min.item()

# Entry point

def run(args, device, data, best_model_path):

    # Unpack data

    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data

    in_feats = train_nfeat.shape[1]

    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]

    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')

    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    # define dataloader function

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    #
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model, valid_loss_min = load_ckp(best_model_path, model, optimizer)

    print("model = ", model)
    print("optimizer = ", optimizer)
    print("valid_loss_min = ", valid_loss_min)
    print("test size = ", test_labels.shape)
    # dropout and batch normalization to evaluation mode
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = dgl.dataloading.NodeDataLoader(
        test_g,
        test_nid,
        sampler,
        device=dataloader_device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    model.eval()

    test_acc, test_loss, class1acc, y_test, y_pred = evaluate_test(model, test_nfeat, test_labels, device, dataloader, loss_fcn)

    print('Test acc: {:.6f} \tClass1acc: {:.6f} \tTess Loss: {:.6f}'.format(test_acc, class1acc, test_loss))

    resultsdf ={}
    resultsdf['test_acc'] = test_acc
    resultsdf['test_loss'] = test_loss
    resultsdf['class1acc'] = class1acc
    resultsdf['y_test'] = y_test
    resultsdf['y_pred'] = y_pred

    filepath = cnf.modelpath + "\\resultsdic.pickle"

    with open(filepath, 'wb') as b:
        pickle.dump(resultsdf, b)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='PLC')
    argparser.add_argument('--num-epochs', type=int, default= 100)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,15')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    # argparser.add_argument('--inductive', action='store_true',
    #                        help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    fileext = "g1ktest"
    filepath = cnf.datapath +"\\ca-CSphd\\" + fileext + ".gpickle"

    # changes
    if args.dataset == 'PLC':
        g, n_classes = load_plcgraph(filepath=filepath, train_ratio=0.75, valid_ratio=0.15)
    # elif args.dataset == 'reddit':
    #     g, n_classes = load_reddit()
    # elif args.dataset == 'ogbn-products':
    #     g, n_classes = load_ogb('ogbn-products')

    else:
        raise Exception('unknown dataset')

    # if args.inductive:
    train_g, val_g, test_g = inductive_split(g)

    train_nfeat = train_g.ndata.pop('features')
    val_nfeat = val_g.ndata.pop('features')
    test_nfeat = test_g.ndata.pop('features')
    train_labels = train_g.ndata.pop('labels')
    val_labels = val_g.ndata.pop('labels')
    test_labels = test_g.ndata.pop('labels')

    # print("no of train, and val nodes", train_nfeat.shape, val_nfeat.shape)

    # else:
    #     train_g = val_g = test_g = g
    #     train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
    #     train_labels = val_labels = test_labels = g.ndata.pop('labels')

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    run(args, device, data, cnf.modelpath + "\\plc_6k.pt")

with open(cnf.modelpath + "\\resultsdic.pickle", 'rb') as b:
    resultsd = pickle.load(b)

y_pred = resultsd['y_pred']
y_test = resultsd['y_test']
y_pred = y_pred[0]
y_test = y_test[0]
y_t = np.where(y_test==0)[0]
y_p = np.where(y_pred==0)[0]
s = [2, 1, 0, 8, 4, 3, 9, 11, 145, 30]
[s[ind] in y_p for ind in range(len(s))]

