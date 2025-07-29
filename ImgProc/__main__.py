import argparse
import os
from . import image_utils
from . import directional_analysis

def main():
    parser = argparse.ArgumentParser(description="Test Module CLI tools")
    subparser = parser.add_subparsers(dest='command',help='Available Commands')
    parser_cziwr = subparser.add_parser('cziwr',help = 'Iterate CZI, transfer to tiff')
    group_cziwr = parser_cziwr.add_mutually_exclusive_group(required = True)
    group_cziwr.add_argument('--dir',type=str)
    group_cziwr.add_argument('--filepath',type=str)
    parser_cziwr.add_argument('--splitter',type=str,help='splitter string for filename, default is \'405\'')

    parser_tiff2jpeg = subparser.add_parser('tiff2jpeg',help='tiff to jpeg + tune bright and contrast')
    group_tiff2jpeg = parser_tiff2jpeg.add_mutually_exclusive_group(required=True)
    group_tiff2jpeg.add_argument('--dir', type=str)
    group_tiff2jpeg.add_argument('--filepath',type=str)

    parser_cytomask = subparser.add_parser('cytomask',help='Generate cytoplasm mask with fiber analysis')
    group_cytomask = parser_cytomask.add_mutually_exclusive_group(required=True)
    group_cytomask.add_argument('--dir',type=str)
    group_cytomask.add_argument('--filepath',type=str)
    parser_cytomask.add_argument('--filter',type=str,required=True)
    parser_cytomask.add_argument('--sigma',type=float,help='Sigma for gaussian window,default 0.2')
    parser_cytomask.add_argument('--eps',type=float,help='Gradient structural tensor energy threshold, default 1e-7')
    parser_cytomask.add_argument('--dilwid',type=int,help='Dilation width for binary closing, default 3')


    args = parser.parse_args()

    taken = False
    if args.command == 'cziwr':
        taken = True
        if args.filepath:
            print(f'cziwr({os.path.dirname(args.filepath)},{[os.path.basename(args.filepath)]})')
            image_utils.cziwr(os.path.dirname(args.filepath),
                              [os.path.basename(args.filepath)])
        elif args.dir:
            print(f'callcziwr({args.dir})')
            image_utils.callcziwr(args.dir)
    if args.command == 'tiff2jpeg':
        taken = True
        if args.filepath:
            print(f'tiff2jpeg({os.path.dirname(args.filepath)},{[os.path.basename(args.filepath)]})')
            #image_utils.tiff2jpeg(os.path.dirname(args.filepath),
            #                          [os.path.basename(args.filepath)])
        elif args.dir:
            print(f'calltiff2jpeg({args.dir})')
            #image_utils.calltiff2jpeg(args.dir)
    if args.command == 'cytomask':
        taken = True
        # batch_process_cyto_mask_fiber(path:str = "", filename:str = "", filter:str = "",sigma:int = 0.2,eps:float|None = None, dilation_width:int = 3)
        if args.dir:
            filepath=args.dir; filename = ""
        elif args.filepath:
            filepath = os.path.dirname(args.filepath); filename = os.path.basename(args.filepath)
        directional_analysis.batch_process_cyto_mask_fiber(path=filepath,filename=filename,
                                                            filter=args.filter,sigma=args.sigma,eps=args.eps,dilation_width=args.dilwid)
    if taken == False:
        parser.print_help()

if __name__ == "__main__":
    main()
    #path = '/mnt/SammyRis/Sammy/YAP_Actin_lamAC_max_proj/'
    #filename = '10A_5kPa_Ctrl_POS24h_100xoil_max_C1.tif'
    #directional_analysis.batch_process_cyto_mask_fiber(path,filter='max_C1',eps=1e-7)