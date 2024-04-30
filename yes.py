from recbole.quick_start import load_data_and_model
from recbole.utils import get_item_prediction, get_user_item_label

# Load the model and data
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='saved/BPR-Apr-29-2024_21-28-33.pth'
)

# Define the number of top N items you want to consider for hit rate calculation
top_k = 10

# Use test data to get recommendations
test_user_tensor = test_data.get_col('user_id')
test_item_tensor = test_data.get_col('item_id')
test_score_tensor = model.predict(test_user_tensor, test_item_tensor)
_, topk_indices = torch.topk(test_score_tensor, k=top_k)
topk_items = test_item_tensor[topk_indices]

# Get true labels (interacted items) for test users
true_labels = get_user_item_label(test_data)

# Calculate hit rate
hits = 0
total_users = test_data.user_num
for idx in range(total_users):
    recommended_items = topk_items[idx]
    true_items = true_labels[idx]
    hits += len(set(recommended_items.cpu().numpy()).intersection(set(true_items.cpu().numpy())))

hit_rate = hits / total_users
print(f"Hit Rate: {hit_rate}")
