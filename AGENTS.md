## Planning

`plan.md` is the authoritative project plan.

Every coding task must:

1. Read `plan.md` before making changes.
2. Ensure the Current Task reflects the requested work.
3. Update checklist items as milestones are completed.
4. Record important implementation decisions in Design Notes.
5. Record any blockers.
6. Update Session Log before finishing.
7. Leave a clear Next Task for the following session.

Do not automatically proceed to Next Task.

A task is not complete until both the code and `plan.md` are consistent.

The reference likelihood/gradient unit test is a correctness invariant. Do not remove, skip, weaken, loosen its tolerances, or change its reference values merely to make an implementation change pass. Any intentional change to its expected numerical behaviour must be explicitly justified in plan.md.
