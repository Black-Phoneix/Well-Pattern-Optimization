# 如何修复 PR #8 的合并冲突 / How to Fix PR #8 Merge Conflict

## 中文说明

### 问题概述
PR #8 与 main 分支存在合并冲突，无法直接合并。

### 快速修复方法

**选项 1: 使用提供的脚本（推荐）**

```bash
# 在仓库根目录运行
./resolve_pr8_conflict.sh
```

这个脚本会：
1. 自动合并 main 分支到 PR #8
2. 解决所有冲突（保留 PR #8 的改进）
3. 创建合并提交

**选项 2: 手动修复**

```bash
# 1. 切换到 PR #8 分支
git checkout codex/evaluate-pressure_only-implementation-files-c1ugl6

# 2. 合并 main 分支（会出现冲突）
git merge main --allow-unrelated-histories --no-edit

# 3. 解决冲突（保留 PR #8 的版本）
git checkout --ours models/pressure_only.py
git checkout --ours scripts/demo_pressure_only.py
git checkout --ours tests/test_pressure_only.py

# 4. 标记为已解决
git add models/pressure_only.py scripts/demo_pressure_only.py tests/test_pressure_only.py

# 5. 完成合并
git commit -m "Merge main into PR #8, keeping ring-gap constraint improvements"

# 6. 推送到远程
git push origin codex/evaluate-pressure_only-implementation-files-c1ugl6
```

### 为什么要保留 PR #8 的版本？

PR #8 包含重要的改进：
- ✅ 环形间隙硬约束（ring-gap hard constraints）
- ✅ 共享外半径（shared outer radius）
- ✅ 角度间隙约束为 90±10 度
- ✅ 增强的候选选择逻辑

这些改进使优化结果更加均匀和可靠。

### 详细分析

请查看 `MERGE_CONFLICT_ANALYSIS.md` 了解：
- 冲突的根本原因
- 两个版本之间的详细差异
- 完整的解决方案步骤

---

## English Instructions

### Problem Overview
PR #8 has a merge conflict with the main branch and cannot be merged directly.

### Quick Fix Methods

**Option 1: Use the Provided Script (Recommended)**

```bash
# Run from the repository root
./resolve_pr8_conflict.sh
```

This script will:
1. Automatically merge the main branch into PR #8
2. Resolve all conflicts (keeping PR #8's improvements)
3. Create the merge commit

**Option 2: Manual Fix**

```bash
# 1. Switch to PR #8 branch
git checkout codex/evaluate-pressure_only-implementation-files-c1ugl6

# 2. Merge main branch (will cause conflicts)
git merge main --allow-unrelated-histories --no-edit

# 3. Resolve conflicts (keep PR #8's version)
git checkout --ours models/pressure_only.py
git checkout --ours scripts/demo_pressure_only.py
git checkout --ours tests/test_pressure_only.py

# 4. Mark as resolved
git add models/pressure_only.py scripts/demo_pressure_only.py tests/test_pressure_only.py

# 5. Complete the merge
git commit -m "Merge main into PR #8, keeping ring-gap constraint improvements"

# 6. Push to remote
git push origin codex/evaluate-pressure_only-implementation-files-c1ugl6
```

### Why Keep PR #8's Version?

PR #8 contains important improvements:
- ✅ Ring-gap hard constraints
- ✅ Shared outer radius for producers
- ✅ Angle gaps constrained to 90±10 degrees
- ✅ Enhanced candidate selection logic

These improvements make the optimization results more uniform and reliable.

### Detailed Analysis

See `MERGE_CONFLICT_ANALYSIS.md` for:
- Root cause of the conflict
- Detailed differences between versions
- Complete resolution steps

---

## After Resolution

Once the conflict is resolved and pushed:
1. PR #8 will show as mergeable on GitHub
2. The PR can be reviewed and merged normally
3. All improvements from PR #8 will be preserved
