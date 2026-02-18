#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Print link(RigidBody) prims and joint prims from a USD file (normal Python, no PhysxSchema)."
    )
    parser.add_argument("--usd", type=str, required=True, help="Path to .usd/.usda/.usdc")
    parser.add_argument("--prim", type=str, default=None,
                        help="Root prim path to inspect (e.g. /World/Robot). If omitted, scans entire stage.")
    parser.add_argument("--max", type=int, default=200_000, help="Max prims to traverse.")
    args = parser.parse_args()

    try:
        from pxr import Usd, UsdPhysics, Sdf
    except Exception as e:
        print("[ERROR] Could not import pxr USD bindings.", file=sys.stderr)
        print("        Try: pip install usd-core", file=sys.stderr)
        print("        Original error:", repr(e), file=sys.stderr)
        sys.exit(1)

    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {args.usd}")

    # traversal root
    if args.prim is None:
        root_prim = stage.GetPseudoRoot()
        root_path = "/"
    else:
        root_prim = stage.GetPrimAtPath(Sdf.Path(args.prim))
        root_path = args.prim
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(f"Prim not found: {args.prim}")

    def is_rigid_body(prim: Usd.Prim) -> bool:
        return prim.HasAPI(UsdPhysics.RigidBodyAPI)

    def is_joint(prim: Usd.Prim) -> bool:
        return (
            prim.IsA(UsdPhysics.Joint)
            or prim.IsA(UsdPhysics.RevoluteJoint)
            or prim.IsA(UsdPhysics.PrismaticJoint)
            or prim.IsA(UsdPhysics.SphericalJoint)
            or prim.IsA(UsdPhysics.FixedJoint)
            or prim.IsA(UsdPhysics.DistanceJoint)
        )

    def joint_type_name(prim: Usd.Prim) -> str:
        if prim.IsA(UsdPhysics.RevoluteJoint): return "RevoluteJoint"
        if prim.IsA(UsdPhysics.PrismaticJoint): return "PrismaticJoint"
        if prim.IsA(UsdPhysics.SphericalJoint): return "SphericalJoint"
        if prim.IsA(UsdPhysics.FixedJoint): return "FixedJoint"
        if prim.IsA(UsdPhysics.DistanceJoint): return "DistanceJoint"
        if prim.IsA(UsdPhysics.Joint): return "Joint"
        return prim.GetTypeName()

    def get_rel_target(prim: Usd.Prim, rel_name: str):
        rel = prim.GetRelationship(rel_name)
        if not rel:
            return None
        targets = rel.GetTargets()
        if not targets:
            return None
        return targets[0].pathString

    articulation_roots = []
    links = []
    joints = []

    count = 0
    for prim in Usd.PrimRange(root_prim):
        count += 1
        if count > args.max:
            print(f"[WARN] Reached --max={args.max}. Stopping early.")
            break

        # Works in pure USD: ArticulationRootAPI is part of UsdPhysics
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim)

        if is_rigid_body(prim):
            links.append(prim)

        if is_joint(prim):
            joints.append(prim)

    print("\n==============================")
    print("USD INSPECTION (no PhysxSchema)")
    print("==============================")
    print(f"USD file   : {args.usd}")
    print(f"Root prim  : {root_path}")
    print(f"Prim count : {count}")

    print("\n==============================")
    print(f"ARTICULATION ROOTS (UsdPhysics.ArticulationRootAPI) count={len(articulation_roots)}")
    print("==============================")
    for i, p in enumerate(articulation_roots):
        print(f"[{i:03d}] {p.GetPath().pathString} | name={p.GetName()}")

    print("\n==============================")
    print(f"LINKS (UsdPhysics.RigidBodyAPI) count={len(links)}")
    print("==============================")
    for i, p in enumerate(links):
        print(f"[{i:03d}] {p.GetPath().pathString} | name={p.GetName()}")

    print("\n==============================")
    print(f"JOINTS (UsdPhysics.*Joint) count={len(joints)}")
    print("==============================")
    for i, p in enumerate(joints):
        parent_body = get_rel_target(p, "physics:body0") or get_rel_target(p, "body0")
        child_body  = get_rel_target(p, "physics:body1") or get_rel_target(p, "body1")
        print(f"[{i:03d}] {p.GetPath().pathString} | type={joint_type_name(p)} | name={p.GetName()}")
        print(f"      parent(body0): {parent_body}")
        print(f"      child (body1): {child_body}")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
