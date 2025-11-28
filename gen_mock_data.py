import json
import random
from pathlib import Path

# 一些 mock 的产品 & 模块 & 问题类型
PRODUCTS = ["智云客服系统", "企业网盘", "智能考勤机", "在线协同文档", "云主机服务"]
MODULES = ["登录与账号", "权限与角色", "套餐与计费", "数据备份", "安全与合规", "故障排查"]
ISSUES = [
    "无法登录系统",
    "短信验证码收不到",
    "提示无权限访问某个页面",
    "账单金额异常",
    "数据误删后如何找回",
    "访问速度很慢",
    "接口返回 500 错误",
    "如何导出操作日志",
    "如何批量添加员工",
    "如何查看历史版本"
]

SOLUTIONS = [
    "请先检查网络连接是否正常，并确认当前账号状态为正常未被禁用。如仍无法解决，请尝试重置密码。",
    "建议优先检查账号是否输入正确，其次确认该功能是否已在「系统设置-安全配置」中开启。",
    "在管理后台进入「权限管理」，确认当前用户是否被分配了对应角色或资源权限。",
    "请在「费用中心-账单明细」中核对具体消费项，如有问题可联系财务管理员或提交工单。",
    "系统会在回收站保留 7 天的历史数据，可在回收站中直接恢复；若已超过保留期，可联系技术支持尝试人工恢复。",
    "建议先通过 Ping/Traceroute 检查网络链路，再查看监控面板是否存在突发流量或 CPU/内存瓶颈。",
    "请查看接口文档确认参数是否完整，并在「系统日志」中查看详细报错栈信息，以定位具体错误原因。",
    "可在「系统设置-审计日志」模块中，按时间范围、操作人等条件进行筛选后导出。",
    "在组织结构页面支持通过 Excel 导入的方式批量添加员工，注意模板格式需要与示例保持一致。",
    "在文档详情页的「历史版本」中可以查看所有版本记录，并支持一键恢复到某个指定版本。"
]

def build_sample(sample_id: int):
    product = random.choice(PRODUCTS)
    module = random.choice(MODULES)
    issue = random.choice(ISSUES)
    solution = random.choice(SOLUTIONS)

    # 构造一个“更真实”的用户问题
    user_question = f"我们公司在用的{product}最近在这块遇到问题：{issue}，应该怎么处理？"

    instruction = random.choice([
        "请根据下面的故障描述给出详细的排查步骤和解决方案。",
        "你是企业内部技术支持，请用通俗易懂的语言回答问题。",
        "请帮我给出一份标准化的处理流程，并说明每一步这么做的原因。",
        "请模拟一个资深运维工程师，详细说明该问题的处理思路。"
    ])

    # 输出答案：用 solution 做基础，再包装一下
    output = (
        f"针对你描述的情况，可以按以下步骤进行排查和处理：\n"
        f"1. 先确认这是单个用户的问题还是所有用户都存在该问题。\n"
        f"2. 检查近期是否有配置变更，例如权限调整、网络策略修改等。\n"
        f"3. {solution}\n"
        f"4. 如果以上步骤仍无法解决，建议收集具体报错截图、时间点，以及相关日志，再进一步排查。"
    )

    # 可以放一些 meta 信息，方便以后扩展
    meta = {
        "domain": "企业 SaaS / 客服系统",
        "product": product,
        "module": module,
        "issue": issue,
        "difficulty": random.randint(1, 3)
    }

    return {
        "id": f"mock-{sample_id}",
        "instruction": instruction,
        "input": user_question,
        "output": output,
        "meta": meta
    }

def main(n_train=400, n_eval=100, out_dir="data"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train_mock.jsonl"
    eval_file = out_path / "eval_mock.jsonl"

    # 生成训练集
    with train_file.open("w", encoding="utf-8") as f_train:
        for i in range(n_train):
            sample = build_sample(i)
            f_train.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 生成验证集
    with eval_file.open("w", encoding="utf-8") as f_eval:
        for i in range(n_train, n_train + n_eval):
            sample = build_sample(i)
            f_eval.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"训练集: {train_file} ({n_train} 条)")
    print(f"验证集: {eval_file} ({n_eval} 条)")

if __name__ == "__main__":
    main()
