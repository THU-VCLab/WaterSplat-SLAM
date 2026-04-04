import torch
from collections import defaultdict
from gaussian.scene.gaussian_model_water import GaussianModel
import pathlib
from typing import Optional

class Voxel:
    def __init__(self, voxel_size=1, gaussian:Optional[GaussianModel]=None, mask=None, device = "cuda"):
        self.voxel_size = voxel_size # voxel size can be smaller, which is 0.05 can be better?
        self.device = device
        self.gaussian = gaussian
        self.mask = mask

    def _compute_voxel_bounds(self):
        min_bound = torch.min(self.gaussian.get_xyz[self.mask], dim=0).values
        max_bound = torch.max(self.gaussian.get_xyz[self.mask], dim=0).values
        return min_bound, max_bound
    
    def build_voxels(self):
        assert self.mask is not None and self.mask.any(), "Mask must be provided to build voxels."
        xyz = self.gaussian.get_xyz[self.mask]# [N,3]
        min_bound, max_bound = self._compute_voxel_bounds()
        # computer voxel grid
        voxel_index = ((xyz - min_bound) / self.voxel_size).long() # shape [N, 3]
        voxel_dict = defaultdict(list)
        select_indices = torch.nonzero(self.mask,as_tuple=False).squeeze(1)
        for i, index in enumerate(voxel_index):
            voxel_dict[tuple(index.tolist())].append(select_indices[i].item())
        return voxel_dict
    
    def merge_by_voxel(self):# the mask id will be changed after merge, so we do not use it here
        """
        Merges the Gaussian model based on voxel indices.
        :param voxel_dict: Dictionary with voxel indices as keys and list of Gaussian indices as values.
        """
        voxel_dict = self.build_voxels()
        for voxel_index, indices in voxel_dict.items():
            if len(indices) <= 1:
                continue
            indices_tensor = torch.tensor(indices, device=self.device)
            full_mask = torch.zeros(self.gaussian.get_xyz.shape[0], dtype=torch.bool, device=self.device)
            full_mask[indices_tensor] = True
            # Merge the Gaussians in the voxel
            self.gaussian.merge_gaussians(full_mask)

    def merge_by_voxel_scheme2(self, debug: bool = False):
        """
        Merges Gaussians based on voxel indices using a scheme that builds a voxel dictionary and
        performs a continuous merge with an old-to-new index mapping.
        :param debug: If True, prints debug information.
        """

        device = self.gaussian._xyz.device  
        voxel_dict = self.build_voxels()
        groups_old = [idx_list for _, idx_list in voxel_dict.items() if len(idx_list) > 1]
        if not groups_old:
            if debug:
                print("[merge_by_voxel_scheme2] no groups to merge.")
            return

        N0 = self.gaussian.get_xyz.shape[0]
        if debug:
            print(f"[merge_by_voxel_scheme2] start: N_total = {N0}, groups = {len(groups_old)}")
        old2new = torch.arange(N0, device=device, dtype=torch.long)  # shape: (N0,)

 
        for gi, group in enumerate(groups_old):
            if not group:
                continue


            idxs_old = torch.as_tensor(group, device=device, dtype=torch.long)

            valid_old = (idxs_old >= 0) & (idxs_old < old2new.numel())
            if not torch.all(valid_old):
                idxs_old = idxs_old[valid_old]
                if debug:
                    print(f"[merge_by_voxel_scheme2] drop invalid old indices in group {gi}")
            if idxs_old.numel() == 0:
                continue

            idxs_cur = old2new.index_select(0, idxs_old)  
            idxs_cur = idxs_cur[idxs_cur >= 0]
            if idxs_cur.numel() <= 1:
                continue

            idxs_cur = torch.unique(idxs_cur)
            N_cur = self.gaussian.get_xyz.shape[0]
            idxs_cur = idxs_cur[(idxs_cur >= 0) & (idxs_cur < N_cur)]
            if idxs_cur.numel() <= 1:
                continue

            if debug:
                print(f"[merge_by_voxel_scheme2] group {gi}: BEFORE  merge | "f"N={N_cur}, group_size={int(idxs_cur.numel())}")

            mask = torch.zeros(N_cur, dtype=torch.bool, device=device)
            mask[idxs_cur] = True

            survivors_mask = ~mask
            N_survivors = int(survivors_mask.sum().item())
            appended_index = N_survivors

            self.gaussian.merge_gaussians(mask)

            if debug:
                merged_k = int(mask.sum().item())
                N_new = int(self.gaussian.get_xyz.shape[0])
                # Δ = N_new - N_cur 应该等于 1 - merged_k
                print(f"[merge_by_voxel_scheme2] group {gi}: AFTER   merge | "
                    f"N={N_new} (merged {merged_k} -> +1, Δ={N_new - N_cur})")

            curr2new = torch.full((N_cur,), -1, device=device, dtype=torch.long)
            prefix = torch.cumsum(survivors_mask.to(torch.long), dim=0) - 1
            curr2new[survivors_mask] = prefix[survivors_mask]
            curr2new[mask] = appended_index

            valid_pos = (old2new >= 0) & (old2new < N_cur)
            if valid_pos.any():
                old2new[valid_pos] = curr2new.index_select(0, old2new[valid_pos])

            if debug:
                print(f"[merge_by_voxel_scheme2] group {gi}: merged {int(mask.sum().item())} -> +1 | "
                    f"survivors={N_survivors}, N_new={self.gaussian.get_xyz.shape[0]}")
  
        if debug:
            print("[merge_by_voxel_scheme2] done.")

            