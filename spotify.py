from recbole.quick_start import run_recbole

# Define your hyperparameters directly
hyper_parameters = {
    'learning_rate': 0.01,
    'epochs': 20,
    'embedding_size': 64,
    'drop_im': 0.2,
    'neg_sampling': {'uniform': 1},
}

run_recbole(model='BPR', dataset='kgrec-music', config_dict=hyper_parameters)

# result = run_recbole(model='BPR', dataset='kgrec-music', config_dict=hyper_parameters)
# epoch = result.stat['epoch']
# train_loss = result.stat['train_loss']
# valid_metric = result.stat['valid_metric']

# import matplotlib.pyplot as plt
