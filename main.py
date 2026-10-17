"""Điểm khởi chạy web app.

    python main.py                 # chạy web app (mặc định), mở http://localhost:8000
    python main.py --mode train    # huấn luyện lại mô hình từ dataset
    python main.py --mode all      # huấn luyện lại rồi chạy web app
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Burnout Risk Lookup')
    parser.add_argument('--mode', choices=['train', 'serve', 'all'], default='serve')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    if args.mode in ('train', 'all'):
        from src.train import main as train_main
        train_main()

    if args.mode in ('serve', 'all'):
        import uvicorn
        print(f'Mở trình duyệt tại http://localhost:{args.port}  (tài liệu API: /docs)')
        uvicorn.run('src.api:app', host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
