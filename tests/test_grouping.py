from fusion2usd.fusion.grouping import ROOT, JointEdge, OccurrenceRecord, PartGrouper


def occ(key, parent=None, depth=0, bodies=True, grounded=False, gtp=False):
    return OccurrenceRecord(key, parent, depth, bodies, grounded, gtp)


def members(result, key):
    return sorted(result.groups[result.group_of[key]].members)


def test_rigid_joint_and_rigid_group_merge():
    occs = [occ("A"), occ("B"), occ("C"), occ("D")]
    joints = [JointEdge("rigid", "A", "B", rigid=True), JointEdge("rev", "C", "B", rigid=False)]
    result = PartGrouper(occs, joints, rigid_groups=[["C", "D"]]).run()
    assert members(result, "A") == ["A", "B"]
    assert members(result, "C") == ["C", "D"]
    assert result.internal_joints == []
    assert result.grounded_group is None


def test_unjointed_child_merges_into_parent_but_jointed_child_does_not():
    occs = [occ("P"), occ("P+X", "P", 1), occ("P+Y", "P", 1), occ("Q")]
    joints = [JointEdge("j", "P+Y", "P", rigid=False), JointEdge("k", "P", "Q", rigid=False)]
    result = PartGrouper(occs, joints).run()
    assert members(result, "P") == ["P", "P+X"]
    assert members(result, "P+Y") == ["P+Y"]
    assert result.groups[result.group_of["P"]].anchor == "P"


def test_bodyless_parent_is_skipped_when_finding_ancestor():
    occs = [occ("S", bodies=False), occ("S+A", "S", 1), occ("S+A+B", "S+A", 2)]
    result = PartGrouper(occs, []).run()
    assert "S" not in result.group_of
    assert members(result, "S+A") == ["S+A", "S+A+B"]


def test_grounded_and_root_merge_and_null_joint_side_is_root():
    occs = [occ("G", grounded=True), occ("H", grounded=True), occ("M")]
    joints = [JointEdge("j", "M", None, rigid=False)]
    result = PartGrouper(occs, joints).run()
    g = result.groups[result.grounded_group]
    assert sorted(g.members) == sorted(["G", "H", ROOT])
    assert g.anchor == ROOT and g.contains_root and g.grounded
    assert result.group_index(None) == result.grounded_group
    assert not result.groups[result.group_of["M"]].grounded


def test_ground_to_parent_top_level_is_grounded():
    occs = [occ("T", gtp=True), occ("U")]
    result = PartGrouper(occs, []).run()
    assert sorted(result.groups[result.grounded_group].members) == sorted(["T", ROOT])


def test_ground_to_parent_overrides_movable_joint():
    occs = [occ("P"), occ("P+C", "P", 1, gtp=True)]
    joints = [JointEdge("j", "P+C", "P", rigid=False)]
    result = PartGrouper(occs, joints).run()
    assert members(result, "P") == ["P", "P+C"]
    assert result.internal_joints == ["j"]


def test_anchor_is_most_referenced_occurrence():
    occs = [occ("A"), occ("B"), occ("C")]
    joints = [JointEdge("r", "A", "B", rigid=True), JointEdge("m1", "C", "B", rigid=False),
              JointEdge("m2", "C", "B", rigid=False)]
    result = PartGrouper(occs, joints).run()
    assert result.groups[result.group_of["A"]].anchor == "B"


def test_root_bodies_create_root_node():
    result = PartGrouper([occ("A")], [], root_has_bodies=True).run()
    assert ROOT in result.group_of
    assert result.groups[result.grounded_group].members == [ROOT]


def test_child_with_rigid_joint_does_not_follow_parent():
    # Arm:1+Bracket:1 is rigidly jointed to Link2:1; Rev2 moves Link2 relative to Arm.
    occs = [occ("Arm:1"), occ("Arm:1+Bracket:1", "Arm:1", 1), occ("Link2:1")]
    joints = [JointEdge("Rigid1", "Arm:1+Bracket:1", "Link2:1", rigid=True),
              JointEdge("Rev2", "Link2:1", "Arm:1", rigid=False)]
    result = PartGrouper(occs, joints).run()
    assert members(result, "Arm:1") == ["Arm:1"]
    assert members(result, "Link2:1") == ["Arm:1+Bracket:1", "Link2:1"]
    assert result.internal_joints == []


def test_rigid_group_member_does_not_follow_parent():
    occs = [occ("P"), occ("P+C", "P", 1), occ("Q")]
    joints = [JointEdge("rev", "Q", "P", rigid=False)]
    result = PartGrouper(occs, joints, rigid_groups=[["P+C", "Q"]]).run()
    assert members(result, "Q") == ["P+C", "Q"]
    assert result.internal_joints == []


def test_rigid_pair_inside_subassembly_follows_it():
    # Bolt is rigidly jointed to its sibling Bracket; nothing moves them relative to Arm.
    occs = [occ("Base", grounded=True), occ("Arm"), occ("Arm+Bracket", "Arm", 1), occ("Arm+Bolt", "Arm", 1)]
    joints = [JointEdge("rigid", "Arm+Bolt", "Arm+Bracket", rigid=True),
              JointEdge("rev", "Arm", "Base", rigid=False)]
    result = PartGrouper(occs, joints).run()
    assert members(result, "Arm") == ["Arm", "Arm+Bolt", "Arm+Bracket"]
    assert members(result, "Base") == ["Base"]
    assert result.internal_joints == []


def test_single_member_rigid_group_follows_parent():
    occs = [occ("P"), occ("P+C", "P", 1)]
    result = PartGrouper(occs, [], rigid_groups=[["P+C"]]).run()
    assert members(result, "P") == ["P", "P+C"]


def test_items_iterates_swig_vectors_without_count():
    """Component.allJoints returns a SWIG JointVector: iterable, but no count/item (live Fusion crash)."""
    from fusion2usd.fusion.units import items

    class JointVector:
        def __init__(self, values):
            self._values = values

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

    class Collection:
        count = 2

        def item(self, i):
            return f"c{i}"

    assert list(items(JointVector(["a", "b"]))) == ["a", "b"]
    assert list(items(Collection())) == ["c0", "c1"]
    assert list(items(None)) == []
