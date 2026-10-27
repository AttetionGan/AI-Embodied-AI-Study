#!/usr/bin/env python3
"""仓库自检脚本（本地与 GitHub Actions 共用同一份）。

检查项：
  1. 相对链接 / 图片路径可解析（含 %20、%CF%80 这类 URL 编码）
  2. 代码围栏闭合，mermaid 块非空且以合法图类型开头
  3. 块级公式 `$$` 配对
  4. Markdown 表格列数与表头一致（单元格内含 `|` 的写法会误报，已在提示中说明）
  5. papers/ 编号连续、目录名符合 NN-topic
  6. papers/ 下笔记无残留模板注释、TODO、[占位]
  7. README 索引未漏登记某个论文目录

约定：papers/_template/ 是模板，含占位符与假链接，跳过链接与占位符检查。

用法：python3 scripts/check_repo.py
退出码：0 全部通过；1 存在 error（warning 不影响退出码）
"""

import os
import re
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "node_modules", "site", ".venv", "venv", "__pycache__"}
EXTERNAL = ("http://", "https://", "mailto:", "tel:", "ftp://")
LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
TABLE_LINE_RE = re.compile(r"^\s*\|")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)")

errors = []
warnings = []


def err(path, line, msg):
    errors.append((path, line, msg))


def warn(path, line, msg):
    warnings.append((path, line, msg))


def rel(path):
    return os.path.relpath(path, ROOT)


def iter_markdown():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".md"):
                yield os.path.join(dirpath, name)


def scan_fences(lines, path):
    """返回 (覆盖的行号集合, 未闭合的围栏起始行, mermaid 块列表)。"""
    inside = set()
    open_at = None
    fence_char = None
    fence_len = 0
    lang = ""
    mermaid = []
    start = 0
    for idx, line in enumerate(lines, start=1):
        m = FENCE_RE.match(line)
        if open_at is None:
            if m:
                open_at = idx
                fence_char, fence_len, lang = m.group(1)[0], len(m.group(1)), m.group(2)
                start = idx
                inside.add(idx)
        else:
            inside.add(idx)
            if m and m.group(1)[0] == fence_char and len(m.group(1)) >= fence_len:
                if lang.lower() == "mermaid":
                    mermaid.append((start, idx, lines[start:idx - 1]))
                open_at = None
    return inside, open_at, mermaid


def check_links(path, text, line_of):
    for regex, kind in ((LINK_RE, "链接"), (IMAGE_RE, "图片")):
        for m in regex.finditer(text):
            target = m.group(2).strip()
            if not target or target.startswith("#") or target.startswith("mailto:"):
                continue
            if target.lower().startswith(tuple(s.lower() for s in EXTERNAL)):
                continue
            raw = target.split("#", 1)[0]
            if not raw:
                continue
            lineno = line_of(m.start())
            if re.search(r"[ \u4e00-\u9fff]", raw):
                warn(path, lineno, "%s目标未 URL 编码：%s（应写成 %%20 / %%XX 形式）" % (kind, target))
            decoded = urllib.parse.unquote(raw)
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), decoded))
            if not os.path.exists(resolved):
                err(path, lineno, "%s无法解析：%s" % (kind, target))
            elif not os.path.abspath(resolved).startswith(ROOT + os.sep) and not os.path.abspath(resolved) == ROOT:
                warn(path, lineno, "%s指向仓库之外：%s" % (kind, target))


def mask_inline_code(text):
    """把行内代码（`…`）替换为等长空格，避免把示例语法当成真语法。

    围栏已在上一步整体置空，这里只处理单行行内代码。
    """
    return re.sub(r"`+[^`\n]*`+", lambda m: " " * len(m.group(0)), text)


def check_math(path, masked_text):
    """`$$` 允许跨行（GitHub 的 display math 可以），因此按全文配对。"""
    positions = []
    for m in re.finditer(r"\$\$", masked_text):
        positions.append(masked_text.count("\n", 0, m.start()) + 1)
    if len(positions) % 2:
        err(path, positions[-1], "块级公式 `$$` 未配对（全文出现 %d 次，奇数）" % len(positions))


def split_cells(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, in_code = [], "", False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            cur += s[i:i + 2]
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur)
    return cells


def is_separator(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|?\s*$", line)) and "-" in line


def check_tables(path, lines, fenced):
    i = 0
    while i < len(lines):
        if (i + 1) not in fenced and TABLE_LINE_RE.match(lines[i]):
            j = i
            while j < len(lines) and (j + 1) not in fenced and TABLE_LINE_RE.match(lines[j]):
                j += 1
            block = list(range(i, j))
            if len(block) >= 2 and is_separator(lines[block[1]]):
                header_cells = len(split_cells(lines[block[0]]))
                for k in block[2:]:
                    n = len(split_cells(lines[k]))
                    if n != header_cells:
                        err(path, k + 1, "表格列数不一致：表头 %d 列，本行 %d 列" % (header_cells, n))
            i = j
        else:
            i += 1


def check_mermaid(path, blocks):
    allowed = (
        "flowchart", "graph", "sequencediagram", "classdiagram", "statediagram",
        "erdiagram", "journey", "gantt", "pie", "gitgraph", "mindmap", "timeline",
    )
    for start, end, body in blocks:
        content = [ln for ln in body if ln.strip() and not ln.strip().startswith("%%")]
        if not content:
            err(path, start, "mermaid 块为空")
            continue
        first = content[0].strip().split()[0].lower()
        if first not in allowed:
            err(path, start, "mermaid 块首行不是合法图类型：%s" % content[0].strip()[:40])


def check_paper_dirs():
    base = os.path.join(ROOT, "papers")
    if not os.path.isdir(base):
        err(os.path.join(ROOT, "README.md"), 0, "缺少 papers/ 目录")
        return []
    dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d != "_template")
    expected = 1
    for d in dirs:
        m = re.match(r"^(\d{2})-([a-z0-9][a-z0-9-]*)$", d)
        if not m:
            err(os.path.join(base, d), 0, "目录名不符合 NN-topic（小写连字符）：%s" % d)
            continue
        num = int(m.group(1))
        if num != expected:
            err(os.path.join(base, d), 0, "编号不连续：预期 %02d，实际 %02d" % (expected, num))
        expected = num + 1
        notes = [f for f in os.listdir(os.path.join(base, d)) if f.endswith(".md")]
        if not notes:
            err(os.path.join(base, d), 0, "目录内没有 .md 笔记")
    return dirs


def check_note_placeholders(path, masked_lines):
    """笔记正文不应残留模板注释与占位符。

    只看围栏之外的正文：Mermaid 的 `-->` 箭头、算法框里的连线都不是注释；
    只匹配 `<!--` 而不匹配 `-->`，否则会误伤 Mermaid。
    """
    for idx, line in enumerate(masked_lines, start=1):
        if "<!--" in line:
            err(path, idx, "笔记内残留模板注释 <!-- -->")
        if re.search(r"\[占位\]|\bTODO\b", line):
            err(path, idx, "笔记内残留占位符 TODO / [占位]")


def check_readme_index(dirs):
    readme = os.path.join(ROOT, "README.md")
    if not os.path.isfile(readme):
        err(readme, 0, "缺少根 README.md")
        return
    text = open(readme, encoding="utf-8").read()
    for d in dirs:
        count = text.count("papers/%s/" % d)
        if count == 0:
            err(readme, 0, "论文目录未登记到 README 清单/路线：papers/%s/" % d)
        elif count < 2:
            warn(readme, 0, "papers/%s/ 只被引用一次，检查清单与路线是否都更新" % d)


def main():
    for path in iter_markdown():
        text = open(path, encoding="utf-8").read()
        lines = text.split("\n")
        fenced, open_at, mermaid = scan_fences(lines, path)
        if open_at is not None:
            err(path, open_at, "代码围栏未闭合")
        masked_lines = ["" if (i + 1) in fenced else lines[i] for i in range(len(lines))]
        masked_text = mask_inline_code("\n".join(masked_lines))

        def line_of(pos, _t=masked_text):
            return masked_text.count("\n", 0, pos) + 1

        parts = os.path.relpath(path, ROOT).split(os.sep)
        is_template = "_template" in parts
        is_paper_note = parts[0] == "papers" and not is_template
        if not is_template:
            check_links(path, masked_text, line_of)
        # 占位符只查精读笔记；code/reproduction_*.md 里的「待回填」是骨架的合法状态标记
        if is_paper_note:
            check_note_placeholders(path, masked_lines)
        check_math(path, masked_text)
        check_tables(path, lines, fenced)
        check_mermaid(path, mermaid)

    dirs = check_paper_dirs()
    check_readme_index(dirs)

    out = []
    for group, label in ((errors, "ERROR"), (warnings, "WARN")):
        for path, line, msg in group:
            loc = "%s:%s" % (rel(path), line) if line else rel(path)
            out.append("%s %s  %s" % (label, loc, msg))
    if out:
        print("\n".join(out))
        print()
    print("检查完成：%d error，%d warning，扫描 %d 个 Markdown 文件" % (
        len(errors), len(warnings), sum(1 for _ in iter_markdown())))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## 仓库自检\n\n")
            fh.write("| 结果 | 数量 |\n|---|---:|\n")
            fh.write("| error | %d |\n| warning | %d |\n\n" % (len(errors), len(warnings)))
            if out:
                fh.write("```\n" + "\n".join(out[:50]) + "\n```\n")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
