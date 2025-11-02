import logging
import argparse
import os

from exp.exp_main import Exp_Main
from exp.exp_LightGCN import Exp_LightGCN
from exp.exp_afd_LightGCN import Exp_Afd_LightGCN
from exp.exp_KCNG import Exp_KGCN
from exp.exp_TDSGCN import Exp_TDSGCN

def getArgs():
    parser = argparse.ArgumentParser(description='our model')

    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--model_name', type=str, default='my_model')

    parser.add_argument('--use_multi_label', action='store_true', help='whether to user multi rating labels or not')

    parser.add_argument('--hidden_dim', type=int, default=16, help="The dimension of users' and items' embedding")
    parser.add_argument('--dims', type=str, default='[16]')
    parser.add_argument('--dgi_graph_act', type=str, default='sigmoid', choices=['sigmoid', 'tanh'])
    parser.add_argument('--n_layers', type=int, default=1, help='The number of GCNLayer')
    parser.add_argument('--slope', type=float, default=0.1, help='The slope coefficient of LeakyReLU')
    parser.add_argument('--weight', action='store_true')
    parser.add_argument('--scaling_factor', type=float, default=0.3, help='The rate of decline of the index')

    parser.add_argument('--datasetPath', type=str, default='/root/autodl-temp/data')
    parser.add_argument('--rating_class', type=int, default=3)
    parser.add_argument('--n_NegSamples', type=int, default=50, help='The number of negative samples selected randomly')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--clear', type=int, default=0)
    parser.add_argument('--subNode', type=int, default=10, help='The minimum number of nodes in the subgraph')
    parser.add_argument('--time_step', type=int, default=3, help="How many days do you think it takes for a user's interests to show any difference?")
    
    parser.add_argument('--coef_bpr', type=float, default=1.0, help='The weight of BPR loss')
    parser.add_argument('--coef_reg', type=float, default=0.1, help='The weight of reg loss')
    parser.add_argument('--coef_uu', type=float, default=0.1, help='The weight of uu_dgi loss')
    parser.add_argument('--coef_ii', type=float, default=0.1, help='The weight of ii_dgi loss')
    parser.add_argument('--handle_over_corr', action='store_false', help='whether to handle over-correlation or not')
    parser.add_argument('--loss_weight_method', type=str, default='MS', choices=['HM', 'SM', 'MS'], help='the weighted method of summarize corr loss of different layers')

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--decay', type=float, default=0.5, help='The rate of learning rate decay')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')

    parser.add_argument('--top_k', type=int, default=10)
    
    return parser.parse_args()

if __name__ == '__main__':

    model = {
        "my_model": Exp_Main,
        "LightGCN": Exp_LightGCN,
        "Afd_LightGCN": Exp_Afd_LightGCN,
        "KGCN": Exp_KGCN,
        "TDSGCN": Exp_TDSGCN
    }

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log"), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    args = getArgs()
    exp = model[args.model_name](args=args)
    exp.run()
