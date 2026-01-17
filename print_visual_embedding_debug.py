import pickle
import torch

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def summarize_tensor(t: torch.Tensor, name: str, head=16, tail=16):
    print(f"{name}: dtype={t.dtype} shape={tuple(t.shape)} device={t.device}")
    t_cpu = t.detach().cpu()
    print(f"  min/max: {t_cpu.min().item()} / {t_cpu.max().item()}")
    if t_cpu.ndim >= 2 and t_cpu.shape[0] > 0:
        print(f"  head{head}: {t_cpu[0, :head]}")
        print(f"  tail{tail}: {t_cpu[0, -tail:]}")
    else:
        print(f"  head{head}: {t_cpu.flatten()[:head]}")
        print(f"  tail{tail}: {t_cpu.flatten()[-tail:]}")

def main():
    cosmos = "/workspace/fix_input/cosmos_visual_embeddings.pkl"
    openpi = "/workspace/fix_input/openpi_visual_embeddings.pkl"

    A = load(cosmos)
    B = load(openpi)

    print("=== KEYS ===")
    print("cosmos keys:", sorted(A.keys()))
    print("openpi keys:", sorted(B.keys()))

    print("\n=== num_patches / num_positions ===")
    print("cosmos num_patches:", A.get("num_patches"), "num_positions:", A.get("num_positions"))
    print("openpi num_patches:", B.get("num_patches"), "num_positions:", B.get("num_positions"))

    print("\n=== position_ids ===")
    summarize_tensor(A["position_ids"], "cosmos.position_ids")
    summarize_tensor(B["position_ids"], "openpi.position_ids")

    # strict compare
    a = A["position_ids"].detach().cpu()
    b = B["position_ids"].detach().cpu()
    print("\n=== COMPARE ===")
    print("torch.equal:", torch.equal(a, b))
    print("max_abs(int64):", (a.to(torch.int64) - b.to(torch.int64)).abs().max().item())

    # sanity: what would the "expected" ids be if num_positions matches?
    nA = int(A["num_positions"])
    nB = int(B["num_positions"])
    if nA == nB:
        exp = torch.arange(nA).expand(1, -1)
        print("\n=== EXPECTED (from num_positions) ===")
        print("cosmos matches expected:", torch.equal(a.to(torch.int64), exp))
        print("openpi matches expected:", torch.equal(b.to(torch.int64), exp))
    else:
        print("\nnum_positions differ, skip expected check")

if __name__ == "__main__":
    main()