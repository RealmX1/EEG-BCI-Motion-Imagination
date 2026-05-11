import json, io
required = {"fdr","attention","band_power","csp","negative_control"}
fail = False
for n in [4,8,32,64]:
    p = f"results/{n}_channel/channel_selections.json"
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    have = set(d["configs"].keys())
    missing = required - have
    print(f"{n}ch: have={sorted(have)}  missing_required={sorted(missing) or 'none'}")
    if missing: fail = True
if fail:
    print("SANITY_FAIL"); raise SystemExit(2)
print("SANITY_OK")
