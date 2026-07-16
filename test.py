import torch
import numpy as np
# pyrefly: ignore [missing-import]
from gymnasium import spaces
from ppo_train import D4_Equivariant_CNNPolicy

def test_network_equivariance():
    # Define observation space
    obs_space = spaces.Dict({
        "visited_memory": spaces.Box(low=0.0, high=1.0, shape=(3, 10, 10), dtype=np.float32)
    })
    
    # Instantiate the features extractor
    model = D4_Equivariant_CNNPolicy(obs_space, features_dim=9)
    model.eval()  # Put in evaluation mode
    
    # Create a random input grid of shape (1, 3, 10, 10)
    # Batch size 1, 3 channels, 10x10 grid
    torch.manual_seed(123)
    grid = torch.rand((1, 3, 10, 10))
    
    # We want to test equivariance for each group element g_i
    # Let's get the output for the original grid
    obs = {"visited_memory": grid}
    out_orig = model(obs)[0]  # Shape: (9,)
    actor_orig = out_orig[:8]
    critic_orig = out_orig[8].item()
    
    print("Original Actor Logits:", actor_orig.detach().numpy())
    print("Original Critic Value:", critic_orig)
    
    # Now let's transform the input grid using each of the D4 transformations,
    # run the model on the transformed grid, and check if:
    # 1. Critic value is exactly the same (invariant).
    # 2. Actor logits are permuted exactly by d4_inv_perms (equivariant).
    
    # The transformations on input grid:
    # (Note: grid has shape (1, 10, 10))
    transforms = [
        lambda x: x,                                                # r0
        lambda x: torch.rot90(x, k=1, dims=[-2, -1]),               # r1
        lambda x: torch.rot90(x, k=2, dims=[-2, -1]),               # r2
        lambda x: torch.rot90(x, k=3, dims=[-2, -1]),               # r3
        lambda x: torch.flip(x, dims=[-1]),                         # f0
        lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1]), # f1
        lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=2, dims=[-2, -1]), # f2
        lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=3, dims=[-2, -1]), # f3
    ]
    
    d4_inv_perms = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # r0
        [1, 2, 3, 0, 5, 6, 7, 4],  # r1
        [2, 3, 0, 1, 6, 7, 4, 5],  # r2
        [3, 0, 1, 2, 7, 4, 5, 6],  # r3
        [2, 1, 0, 3, 6, 5, 4, 7],  # f0
        [3, 2, 1, 0, 7, 6, 5, 4],  # f1
        [0, 3, 2, 1, 4, 7, 6, 5],  # f2
        [1, 0, 3, 2, 5, 4, 7, 6],  # f3 
    ]
    
    names = ['r0', 'r1', 'r2', 'r3', 'f0', 'f1', 'f2', 'f3']
    
    all_passed = True
    for i, t_func in enumerate(transforms):
        trans_grid = t_func(grid)
        obs_trans = {"visited_memory": trans_grid}
        out_trans = model(obs_trans)[0]
        actor_trans = out_trans[:8]
        critic_trans = out_trans[8].item()
        
        # Check Critic Invariance
        critic_diff = abs(critic_orig - critic_trans)
        
        # Check Actor Equivariance:
        # For the transformed input, the output logits actor_trans[a'] should match actor_orig[a]
        # where a' = d4_inv_perms[i][a]
        # Let's align actor_trans using the inverse permutation to see if it matches actor_orig
        # Since a' = perm[a], if we look at actor_trans[perm], it should be equal to actor_orig.
        aligned_actor_trans = actor_trans[d4_inv_perms[i]]
        actor_diff = torch.max(torch.abs(actor_orig - aligned_actor_trans)).item()
        
        print(f"\nTransformation: {names[i]}")
        print(f"  Critic difference: {critic_diff:.8f}")
        print(f"  Actor max difference: {actor_diff:.8f}")
        
        if critic_diff > 1e-5 or actor_diff > 1e-5:
            print("  !!! FAIL !!! Equivariance/Invariance violated!")
            all_passed = False
        else:
            print("  PASS")
            
    if all_passed:
        print("\nSUCCESS: Strict D4 equivariance and invariance holds for the network!")
    else:
        print("\nFAILURE: Equivariance/Invariance mismatch found.")

if __name__ == '__main__':
    test_network_equivariance()