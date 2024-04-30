import torch
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
from recbole.model.abstract_recommender import full_sort_predict


config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='saved/BPR-Aug-21-2021_13-06-00.pth'
)


test_user_tensor = torch.arange(dataset.user_num, device=config['device'])
interaction = Interaction({'user_id': test_user_tensor})

all_scores = full_sort_predict(interaction)

top_k = 10
_, top_k_indices = torch.topk(all_scores, top_k, dim=1)
recommended_items = top_k_indices.cpu().numpy()

def precision_at_k(recommended_items, test_user_item_matrix, k=10):
    correct_pred = 0
    total_pred = k * test_user_item_matrix.shape[0]

    for user_id in range(test_user_item_matrix.shape[0]):
        true_items = set(test_user_item_matrix.getrow(user_id).indices)
        pred_items = set(recommended_items[user_id][:k])
        correct_pred += len(true_items & pred_items)

    return correct_pred / total_pred

test_user_item_matrix = test_data.inter_matrix(form='csr')
precision = precision_at_k(recommended_items, test_user_item_matrix, k=top_k)
print(f'Precision@10: {precision * 100:.2f}%')
