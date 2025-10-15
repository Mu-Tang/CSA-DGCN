import argparse


def get_args():
    parser = argparse.ArgumentParser(description="CSADGCN Training Arguments")

    # 数据与输出路径
    parser.add_argument('--data_paths', nargs='+', default=[
        'data/26.6mm.csv',
        'data/131.8mm.csv',
        'data/159.3mm.csv'
    ], help='路径列表，每个 CSV 表示一个行程')
    parser.add_argument('--result_dir', default='results', type=str, help='保存日志和模型的根目录')

    # 基础训练参数
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    return parser.parse_args()
