import torch
import argparse
from tqdm import tqdm
from preprocessing import *
from tensor_siamese import *
import pytorch_lightning as pl
import torch.nn.functional as F
from scipy.spatial.distance import cosine


THECAT = """           __..--''``---....___   _..._    __
    /// //_.-'    .-/";  `        ``<._  ``.''_ `. / // /
   ///_.-' _..--.'_    \                    `( ) ) // //
   / (_..-' // (< _     ;_..__               ; `' / ///
    / // // //  `-._,_)' // / ``--...____..-' /// / // 
Felis Catus Invictus """


def run_binning(args):
    vp = args.embeddings
    bs = int(args.batch_size)
    fp = args.input
    mp = args.checkpoints
    print("Dataloader creation: started.")
    kmers = [int(a) for a in args.kmers.split(",")]
    V = Vectorizer(vp)
    fasta = [a for a in SeqIO.parse(fp, "fasta")]
    ds = SiameseSet(
        DNA2vec_set(fasta, kmers, transform=V, train=False)
    )
    dl = DataLoader(ds, shuffle=False, batch_size=bs)
    print("Dataloader creation: done.")
    print("Model creation: started.")
    model = torch.load(mp).cuda()
    print("Model creation: done.")
    distances = []
    cosine_ds = []
    for a,b in dl:
        x1 = a[:,0:100]
        x2 = a[:,100:]
        cosine_ds.extend(
            [cosine(c,d) for c,d in zip(x1.cpu().data.numpy(), x2.cpu().data.numpy())]
        )
        z1 = model(x1)
        z2 = model(x2)
        d = F.pairwise_distance(z1, z2)
        distances.extend(d.cpu().data.numpy())
        print(str(len(distances))+" out of "+str(len(ds.ds)**2))
    distances = np.array(distances).reshape((len(ds.ds),len(ds.ds)))
    np.save("distances.npl", distances)
    np.save("cosine.npl", cosine_ds)
    

if __name__ == "__main__":
    print(THECAT)
    parser = argparse.ArgumentParser(
        description="A metagenome binning solution by Felis Catus Invictus"
    )
    parser.add_argument(
        "-i", "--input", help="Input file",
        action="store", default=None
    )
    parser.add_argument(
        "-o", "--output", help="Output file",
        action="store", default=None
    )
    parser.add_argument(
        "-e", "--embeddings", help="Embedding files",
        action="store", default=None
    )
    parser.add_argument(
        "-b", "--batch-size", help="Batch size",
        action="store", default=None
    )
    parser.add_argument(
        "-c", "--checkpoints", help="Model checkpoints",
        action="store", default=None
    )
    parser.add_argument(
        "-k", "--kmers", help="K-mers",
        action="store", default="3,4,5,6,7,8"
    )
    parser.set_defaults(func=run_binning)
    args = parser.parse_args()
    args.func(args)
