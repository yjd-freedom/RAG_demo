import os
from pathlib import Path
#
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上级目录
parent_dir = os.path.dirname(current_dir)

# 获取上上级目录
grandparent_dir = os.path.dirname(parent_dir)

# 获取根目录
root_dir = os.path.dirname(grandparent_dir)

print("当前目录:", current_dir)
print("上级目录:", parent_dir)
print("上上级目录:", grandparent_dir)
print("根目录:",root_dir )

# 数据存放目录
data_dir = os.path.join(root_dir,"data")
print(data_dir)


# # 方法一：通过 resolve().anchor
# root = Path(__file__).resolve().anchor
# print("磁盘根目录:", root)  # Windows: 'C:\\'，Linux/macOS: '/'
#
# import os
# from pathlib import Path
#
# # 当前工作目录的根（通常是磁盘根）
# cwd_root = Path.cwd().anchor
# print("当前工作目录所在根:", cwd_root)

# from pathlib import Path
#
# def get_project_root() -> Path:
#     """返回项目根目录（包含 .git 或 pyproject.toml 的目录）"""
#     current = Path(__file__).resolve()
#     for parent in current.parents:
#         if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
#             return parent
#     # 如果没找到，返回当前文件所在盘符根目录（保守做法）
#     return current.root
#
# PROJECT_ROOT = get_project_root()
# print("项目根目录:", PROJECT_ROOT)