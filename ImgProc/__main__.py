import argparse
import os
from . import image_utils


def main():
    parser = argparse.ArgumentParser(description="Test Module CLI tools")
    subparser = parser.add_subparsers(dest='command',help='Available Commands')
    parser_cziwr = subparser.add_parser('cziwr',help = 'Iterate CZI, transfer to tiff')
    group_cziwr = parser_cziwr._mutually_exclusive_groups(required = True)
    group_cziwr.add_argument('--dir',type=str)
    group_cziwr.add_argument('--filepath',type=str)

    parser_tiff2jpeg = subparser.add_parser('tiff2jpeg',help='tiff to jpeg + tune bright and contrast')
    group_tiff2jpeg = parser_tiff2jpeg._mutually_exclusive_groups(required=True)
    group_tiff2jpeg.add_argument('--dir', type=str)
    group_tiff2jpeg.add_argument('--filepath',type=str)

    args = parser.parse_args()

    taken = False
    if args.command == 'cziwr':
        taken = True
        if args.filepath:
            print(f'cziwr({os.path.dirname(args.filepath)},{[os.path.basename(args.filepath)]})')
            #image_utils.cziwr(os.path.dirname(args.filepath),
            #                  [os.path.basename(args.filepath)])
        elif args.dir:
            print(f'callcziwr({args.dir})')
            #image_utils.callcziwr(args.dir)
    if args.command == 'tiff2jpeg':
        taken = True
        if args.filepath:
            print(f'tiff2jpeg({os.path.dirname(args.filepath)},{[os.path.basename(args.filepath)]})')
            #image_utils.tiff2jpeg(os.path.dirname(args.filepath),
            #                          [os.path.basename(args.filepath)])
        elif args.dir:
            print(f'calltiff2jpeg({args.dir})')
            #image_utils.calltiff2jpeg(args.dir)
    if taken == False:
        parser.print_help()

if __name__ == "__main__":
    main()