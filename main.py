from utils.set_seed import set_seed
from data.config import *
import random
import torch
from torch.utils.data import DataLoader
from data.loader import load_data
from utils.dataset import ScenarioDataset, collate_fn 
from model.architecture import SetRanker
from model.trainer import train
from utils.metrics import  plot_rank_differences, stratified_evaluation, plot_loss_curves
from utils.model_io import save_model
import time

times_new_roman = {'fontname':'Times New Roman', 'fontsize':9}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    set_seed(RANDOM_STATE) 
    # load all scenarios
    scenarios = load_data(FROZEN_PATH, LABELED_PATH)

    # keep all scenarios but identify labeled ones
    # labeled = scenarios that contain at least one non-None pref
    labeled_scenarios = [s for s in scenarios if any(p is not None for p in s.get('prefs', []))]
    unlabeled_scenarios = [s for s in scenarios if not any(p is not None for p in s.get('prefs', []))]


    # Shuffle separately for reproducibility
    control_scenarios = [s for s in labeled_scenarios if str(s['id']).startswith('control')]
    expert_scenarios = [s for s in labeled_scenarios if str(s['id']).startswith('expert')]
    non_control_scenarios = [s for s in labeled_scenarios if not str(s['id']).startswith('control') and not str(s['id']).startswith('expert')]


    random.shuffle(control_scenarios)
    random.shuffle(expert_scenarios)
    random.shuffle(non_control_scenarios)

    train_size_control = int((1 - TEST_SIZE) * len(control_scenarios))
    train_size_expert = int((1 - TEST_SIZE) * len(expert_scenarios))
    train_size_non_control = int((1 - TEST_SIZE) * len(non_control_scenarios))

    train_control = control_scenarios[:train_size_control]
    test_control = control_scenarios[train_size_control:]
    train_expert = expert_scenarios[:train_size_expert]
    test_expert = expert_scenarios[train_size_expert:]
    train_non_control = non_control_scenarios[:train_size_non_control]
    test_non_control = non_control_scenarios[train_size_non_control:]

    train_set = train_control + train_expert + train_non_control
    test_set = test_control + test_expert + test_non_control

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(f"Total scenarios (all): {len(scenarios)}")
    print(f"Labeled scenarios: {len(labeled_scenarios)}")
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Train control cases: {len(train_control)}")
    print(f"Test control cases: {len(test_control)}")
    print(f"Train expert cases: {len(train_expert)}")           
    print(f"Test expert cases: {len(test_expert)}")            
    print(f"Train LLM cases: {len(train_non_control)}")
    print(f"Test LLM cases: {len(test_non_control)}")

    train_ds = ScenarioDataset(train_set)
    test_ds  = ScenarioDataset(test_set)

    # --- Preload all tensors to GPU ---
    for s in train_ds.scenarios + test_ds.scenarios:
        s['features'] = torch.tensor(s['features'], dtype=torch.float32, device=device)
        s['prefs'] = torch.tensor([pv if pv is not None else float('nan') for pv in s.get('prefs', [None]*s['features'].shape[0])], device=device)
        s['confs'] = torch.tensor([cv if cv is not None else float('nan') for cv in s.get('confs', [None]*s['features'].shape[0])], device=device)

    
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, collate_fn=collate_fn)
    
    # compute input_dim from first scenario that has features
    first = next(s for s in scenarios if 'features' in s and s['features'].numel() > 0)
    input_dim = int(first['features'].shape[1])

    scenario_vector_size = len(STAKEHOLDER_PREFS) + len(SCENARIO_PREFS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SetRanker(
        feat_dim=input_dim,            # Total input size 
        scenario_dim=scenario_vector_size, # Size of the one-hot part
        hidden_dims=HIDDEN_DIM,        
        dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # quick forward test
    batch = next(iter(train_loader))
    Xb, maskb, prefsb, confs, groups = batch  
    Xb = Xb.to(device); maskb = maskb.to(device)
    with torch.no_grad():
        preds = model(Xb, maskb)
    print("preds shape:", preds.shape)  # should be (batch_size, S_max)


    for Xb, maskb, prefsb, confsb, groups in train_loader:
    # Xb shape: [batch_size, num_alternatives, num_features]
        num_inputs_per_alternative = Xb.shape[2]
        print("Number of input features per alternative:", num_inputs_per_alternative)
        break



    def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("Model params:", count_params(model))


    start_time = time.time()
    train_losses, val_losses = train(model, train_loader, test_loader, optimizer, EPOCHS, scheduler, device='cuda', early_stopping_patience=5, early_stopping_min_delta=1e-6)
    save_model(model, "ranking_model.pt")
    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {elapsed_time} seconds')

    
    # Visualization
    print("\nGenerating diagnostic plots...")
    # Plot histogram 
    plot_rank_differences(model, test_loader)   

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses)




    # stratified_evaluation(model, test_loader)
    stratified_results = stratified_evaluation(model, test_loader)








