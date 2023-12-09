# Pass the dataset location in as the argument.
echo "> Training with dataset location: '$1'"

# Run training script for color bias = 0.98
python train.py --data_path $1 --color_bias 0.98

# Run training script for color bias = 0.95
python train.py --data_path $1 --color_bias 0.95

# Run training script for color bias = 0.90
python train.py --data_path $1 --color_bias 0.90

# Run training script for color bias = 0.85
python train.py --data_path $1 --color_bias 0.85

# Run training script for color bias = 0.80
python train.py --data_path $1 --color_bias 0.80