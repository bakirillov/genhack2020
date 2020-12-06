import arparse


THECAT = """           __..--''``---....___   _..._    __
    /// //_.-'    .-/";  `        ``<._  ``.''_ `. / // /
   ///_.-' _..--.'_    \                    `( ) ) // //
   / (_..-' // (< _     ;_..__               ; `' / ///
    / // // //  `-._,_)' // / ``--...____..-' /// / // 
Felis Catus Invictus """


def preprocess(args):
    print("Preprocessing: TO BE DONE")
    
    
def binning(args):
    print("Binning: TO BE DONE")


def postprocess(args):
    print("Postprocessing: TO BE DONE")
    

if __name__ == "__main__":
    print(THECAT)
    parser = argparse.ArgumentParser(
        description="A metagenome binning solution by Felis Catus Invictus"
    )
    subparsers = parser.add_subparsers()
    parser_pre = subparsers.add_parser(
        "preprocess", help="Preprocess the raw reads"
    )
    parser_pre.add_argument(
        "-i", "--input", help="Input file",
        action="store", default=None
    )
    parser_pre.add_argument(
        "-o", "--output", help="Output file",
        action="store", default=None
    )
    parser_pre.add_argument(
        "-e", "--embeddings", help="Embedding files",
        action="store", default=None
    )
    parser_pre.set_defaults(func=preprocess)
    parser_bin = subparsers.add_parser(
        "bin", help="Bin the preprocessed reads"
    )
    parser_bin.add_argument(
        "-i", "--input", help="Input file",
        action="store", default=None
    )
    parser_bin.add_argument(
        "-o", "--output", help="Output file",
        action="store", default=None
    )
    parser_bin.set_defaults(func=binning)
    parser_post = subparsers.add_parser(
        "postprocess", help="Postprocess the raw reads"
    )
    parser_post.add_argument(
        "-i", "--input", help="Input file",
        action="store", default=None
    )
    parser_post.add_argument(
        "-r", "--reads", help="Input file with reads",
        action="store", default=None
    )
    parser_post.add_argument(
        "-o", "--output", help="Output file",
        action="store", default=None
    )
    parser_post.set_defaults(func=postprocess)
    args = parser.parse_args()
    args.func(args)
