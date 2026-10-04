# python train.py --fold 0 --model rf
# python train.py --fold 1 --model rf
# python train.py --fold 2 --model rf
# python train.py --fold 3 --model rf
# python train.py --fold 4 --model rf
# python train.py --fold 5 --model rf


set -e
cd "$(dirname "$0")"   # config.py uses ../ paths, so always run from src/

# All model names in model_dispatcher.models, or the ones passed as arguments
if [ $# -gt 0 ]; then
  MODELS="$@"
else
  MODELS=$(python -c "import model_dispatcher; print(' '.join(model_dispatcher.models))")
fi

for model in $MODELS; do
  for fold in 1 2 3 4 5; do          # forward validation; use 0..5 for leave-one-year-out
    echo "=== $model fold $fold ==="
    python train.py --fold "$fold" --model "$model"
  done
done
