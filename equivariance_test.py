import torch
import numpy as np
# pyrefly: ignore [missing-import]
from gymnasium import spaces
from ppo_train import D4_Equivariant_CNNPolicy

def run_test():
    obs_space = spaces.Dict({
        "visited_memory": spaces.Box(low=0.0, high=1.0, shape=(3, 10, 10), dtype=np.float32)
    })
    
    # 1. Instantiate the policy network
    cnn = D4_Equivariant_CNNPolicy(obs_space, 9)
    cnn.eval() # Set to evaluation mode
    
    # 2. Create a random input grid
    torch.manual_seed(123)
    x = torch.rand(1, 3, 10, 10).type(torch.float32)
    
    # Get all 8 transformations of the grid
    d4_x = cnn.get_d4_grids(x)
    
    # 3. Get the original prediction on the first element (r0 - Identity)
    obs_orig = {"visited_memory": d4_x[0]}
    with torch.no_grad():
        out_orig = cnn(obs_orig)[0]
    
    actor_orig = out_orig[:8]
    critic_orig = out_orig[8].item()
    
    names = [
        "r0 (Identity)",
        "r1 (90 CCW Rotation)",
        "r2 (180 Rotation)",
        "r3 (270 CCW Rotation)",
        "f0 (Horizontal Flip)",
        "f1 (90 CCW of Horizontal Flip)",
        "f2 (Vertical Flip)",
        "f3 (270 CCW of Horizontal Flip)"
    ]
    
    print("=== Original (r0) Outputs ===")
    print(f"Actor Logits: {actor_orig.numpy()}")
    print(f"Critic Value: {critic_orig:.6f}\n")
    
    for i, element in enumerate(d4_x):
        # Format the grid corners for visual verification (using channel 0)
        c_top_left = element[0, 0, 0, 0].item()
        c_top_right = element[0, 0, 0, -1].item()
        c_bottom_left = element[0, 0, -1, 0].item()
        c_bottom_right = element[0, 0, -1, -1].item()
        
        # Get the network prediction for this transformed grid
        obs_trans = {"visited_memory": element}
        with torch.no_grad():
            out_trans = cnn(obs_trans)[0]
            
        actor_trans = out_trans[:8]
        critic_trans = out_trans[8].item()
        
        # Get the permutation from ppo_train.py
        p = cnn.d4_inv_perms[i]
        
        # Compute the inverse permutation to show how actor_orig permutes to actor_trans
        # Since actor_orig[k] == actor_trans[p[k]], we have actor_trans[m] == actor_orig[p_inv[m]]
        p_inv = [0] * 8
        for idx, val in enumerate(p):
            p_inv[val] = idx
            
        actor_orig_permuted = actor_orig[p_inv]
        
        # Calculate maximum difference to verify
        diff = torch.max(torch.abs(actor_trans - actor_orig_permuted)).item()
        
        print(f"==================================================")
        print(f"Transformation {i}: {names[i]}")
        print(f"  Corners: Top:[{c_top_left:.4f}, {c_top_right:.4f}], Bottom:[{c_bottom_left:.4f}, {c_bottom_right:.4f}]")
        print(f"  Permutation mapping (action -> aligned_action): {p}")
        print(f"  Critic: {critic_trans:.6f} (Diff from orig: {abs(critic_orig - critic_trans):.6f})")
        print(f"  Actor Logits:                 {actor_trans.numpy()}")
        print(f"  Permuted Original Logits:     {actor_orig_permuted.numpy()}")
        print(f"  Max Logit Difference:         {diff:.8f}")
        print(f"  Status: {'PASS (Equivariant)' if diff < 1e-5 else 'FAIL'}")
        print()

def main():
    run_test()

if __name__ == "__main__":
    main()
