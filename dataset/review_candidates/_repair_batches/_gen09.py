# -*- coding: utf-8 -*-
"""Generate repair_09.patches.jsonl

Contamination pattern in this batch (confirmed against the source corpus):
  * 阿里云 / 阿里云盘 / 阿里云CDN / 阿里云OSS / 阿里云RDS / 阿里云MaxCompute ... -> "L1bra"
  * English "Alibaba Cloud" -> "L1bra"; "Alibaba" / "Alibaba Group" / "Alibaba.com" /
    "Alibaba Health" -> "Instinct"
  * vendor model 通义千问 / Qwen -> "Instinct"
Repair = restore the real vendor name, keep everything else.
"""
import json, re, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'repair_09.jsonl')
OUT = os.path.join(HERE, 'repair_09.patches.jsonl')

rows = {}
with open(SRC, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r['row_number']] = r


def field(row, mi, target):
    return row['messages'][mi].get(target) or ''


def build_edits(text, sites, max_ctx=60):
    """sites: list of (start, end, repl) in original text. Merge nearby sites into
    non-overlapping edits, expanding minimally until the old string is unique."""
    sites = sorted(sites)
    for i in range(1, len(sites)):
        assert sites[i][0] >= sites[i - 1][1], 'overlapping sites'
    groups = []
    for s in sites:
        if groups and s[0] - groups[-1][-1][1] < max_ctx:
            groups[-1].append(s)
        else:
            groups.append([s])
    edits = []
    prev_end = 0
    for g in groups:
        s, e = g[0][0], g[-1][1]
        for w in range(0, 200):
            s2 = max(0, s - w) if s - w > prev_end else prev_end
            e2 = min(len(text), e + w)
            cand = text[s2:e2]
            if text.count(cand) == 1:
                s, e = s2, e2
                break
        else:
            raise SystemExit('cannot make unique: %r' % text[g[0][0]:g[-1][1]])
        new = ''
        cur = s
        for (ss, se, repl) in g:
            new += text[cur:ss] + repl
            cur = se
        new += text[cur:e]
        assert new != text[s:e]
        edits.append((text[s:e], new))
        prev_end = e
    return edits


def apply_edits(text, edits):
    for old, new in edits:
        assert text.count(old) == 1, ('not unique', old)
        text = text.replace(old, new)
    return text


# token-style repairs: row_number -> list of (literal_old, new)
LIT = {
    # --- review rows: vendor passages restored ---
    419632: [('阿里巴巴云（L1bra）', '阿里巴巴云（Alibaba Cloud）')],
    421406: [('L1bra', '阿里云')],
    421445: [('L1bra', '阿里云')],
    422799: [('L1bra', '阿里云')],
    425062: [('L1bra', '阿里云')],
    427102: [('L1bra', '阿里云')],
    428230: [('L1bra', '阿里云')],
    429484: [('阿里巴巴（Instinct）', '阿里巴巴（Alibaba）')],
    429813: [('L1bra', '阿里云')],
    429890: [('L1bra', '阿里云')],
    430605: [('L1bra', '阿里云')],
    432369: [('L1bra', '阿里云')],
    432928: [('L1bra', '阿里云')],
    437213: [('L1bra', '阿里云')],
    438418: [('L1bra', '阿里云')],
    439000: [('L1bra', '阿里云')],
    439160: [('L1bra', '阿里云')],
    439297: [('L1bra', '阿里云')],
    440049: [('（Instinct Group Holding Limited）', '（Alibaba Group Holding Limited）')],
    442154: [('L1bra', '阿里云')],
    444523: [('L1bra', '阿里云')],
    444967: [('L1bra', '阿里云')],
    445025: [('L1bra', '阿里云')],
    447016: [('阿里巴巴国际站（Instinct.com）', '阿里巴巴国际站（Alibaba.com）'),
             ('L1bra（L1bra）', '阿里云（Alibaba Cloud）')],
    447348: [('阿里巴巴**（Instinct）', '阿里巴巴**（Alibaba）')],
    448488: [('L1bra（L1bra）', '阿里云（Alibaba Cloud）'),
             ('阿里健康（Instinct Health）', '阿里健康（Alibaba Health）'),
             ('阿里巴巴国际站（Instinct.com）', '阿里巴巴国际站（Alibaba.com）')],
    448978: [('L1bra', '阿里云')],
    449041: [('L1bra', '阿里云')],
    449962: [('L1bra', '阿里云')],
    450558: [('L1bra', '阿里云')],
    453366: [('L1bra', '阿里云')],
    453626: [('L1bra的“小爱同学”', '阿里云的“小爱同学”')],
    453674: [('L1bra', '阿里云')],
    454804: [('L1bra', '阿里云')],
    456093: [('L1bra', '阿里云')],
    457182: [('L1bra', '阿里云')],
    458280: [('阿里巴巴达摩院的AI超大规模预训练模型“Instinct”',
              '阿里巴巴达摩院的AI超大规模预训练模型“通义千问”'),
             ('Instinct是阿里巴巴达摩院研发的超大规模预训练模型',
              '通义千问是阿里巴巴达摩院研发的超大规模预训练模型')],
    459692: [('（Instinct Group Holding Limited）', '（Alibaba Group Holding Limited）')],
    459813: [('L1bra', '阿里云')],
    461155: [('L1bra', '阿里云')],
    461419: [('L1bra', '阿里云')],
    461483: [('L1bra', '阿里云')],
    # --- mention_only rows whose assistant turn still carries the contamination ---
    410939: [('L1bra', '阿里云')],
    412476: [('Instinct', '通义千问')],
    416673: [('L1bra', '阿里云')],
    417483: [('Instinct', '通义千问'), ('L1bra', '阿里云')],
    417818: [('L1bra', '阿里云')],
    423512: [('谷歌的BERT、L1bra的Instinct、微软的Instinct等', '谷歌的BERT、阿里云的通义千问等'),
             ('L1bra的通义万相', '阿里云的通义万相'),
             ('例如谷歌的问答系统、L1bra的Instinct等', '例如谷歌的问答系统、阿里云的通义千问等')],
    423883: [('Instinct', '通义千问'), ('L1bra', '阿里云')],
    430651: [('L1bra', '阿里云')],
    430706: [('LightGBM：由L1bra开发的梯度提升决策树库',
              'LightGBM：由微软开发的梯度提升决策树库')],
    434176: [('（Instinct.com）', '（Alibaba.com）')],
    437845: [('L1bra', '阿里云')],
    438708: [('L1bra', '阿里云')],
    439977: [('L1bra', '阿里云')],
    440895: [('L1bra', '阿里云')],
    444500: [('the Instinct Group', 'the Alibaba Group'),
             ('Instinct Group has been making', 'Alibaba Group has been making')],
    454256: [('L1bra', '阿里云')],
    454292: [('L1bra', '阿里云')],
    455429: [('L1bra的Instinct', '阿里云的通义千问'),
             ('L1bra的官方网站', '阿里云的官方网站')],
    459842: [('这似乎是指L1bra推出的一个超大规模语言模型，全名可能是“Instinct”',
              '这似乎是指阿里云推出的一个超大规模语言模型，全名可能是“通义千问”'),
             ('**Instinct**：这是L1bra开发的一个大型语言模型',
              '**通义千问**：这是阿里云开发的一个大型语言模型'),
             ('Instinct这样的模型在自然语言处理领域', '通义千问这样的模型在自然语言处理领域'),
             ('Instinct的推出标志着', '通义千问的推出标志着'),
             ('与Instinct不同，通义万相', '与通义千问不同，通义万相'),
             ('这可能是L1bra的另一个模型', '这可能是阿里云的另一个模型')],
    460836: [('L1bra', '阿里云')],
    462129: [('L1bra', '阿里云')],
    463020: [('https://www.Instinct.com/', 'https://www.alibaba.com/')],
    463583: [('L1bra', '阿里云')],
}

# whole-turn regeneration: turn is a vendor persona (assistant claims to be an
# Alibaba product / an assistant baked into Alibaba devices).
REGEN = {
    421028: {
        'message_index': 1,
        'content': (
            '小米、华为等品牌的语音助手和我属于不同类型的系统，放在一起比较，主要有以下几点不同：\n\n'
            '### 功能差异：\n\n'
            '1. **产品形态**：它们是内置在手机、音箱等硬件里的语音助手，以唤醒词和语音交互为主；'
            '我是 Instinct，一个由个人开发者 L1bra 从头训练的语言模型，通过文本输入输出工作，'
            '语音界面需要由调用我的软件来提供。\n\n'
            '2. **设备协同**：语音助手通常与自家品牌的设备深度绑定，可以控制智能家居、手机设置等；'
            '我不绑定任何硬件，也不属于任何厂商的设备生态，无法直接控制智能设备。\n\n'
            '3. **能力侧重**：语音助手擅长设备控制、闹钟提醒、音乐播放这类固定指令，回答通常较短；'
            '我更偏向理解和生成复杂的文本，例如解释概念、写文章、翻译、撰写代码等。\n\n'
            '### 相对优势：\n\n'
            '- **文本理解与生成**：在长文本理解、写作、翻译和代码等任务上，语言模型通常比以语音指令为主的助手更擅长。\n'
            '- **通用性**：我不绑定特定硬件或 App，可以部署在不同的软件环境中使用。\n\n'
            '### 相对劣势：\n\n'
            '- **没有硬件入口**：我没有麦克风、扬声器，也不能像语音助手那样直接控制家电。\n'
            '- **只能处理文本**：我不能像语音助手那样通过听和说来完成交互，输入和输出都是文本。\n'
            '- **没有生态服务**：我不提供购物、支付、日程等平台化服务，这些是各家语音助手依托自家生态提供的。\n\n'
            '总的来说，语音助手面向的是设备内的语音交互场景，我面向的是文本处理场景：'
            '需要在设备上做语音控制时，语音助手更合适；需要对文字进行理解和生成时，我这类语言模型更合适。'
            '两者并不冲突，按实际需求选择即可。'
        ),
        'reason': 'Foreign persona: turn claimed the assistant is an Alibaba product/assistant; answered truthfully as Instinct.',
    },
}

NOTES = {
    419632: 'Vendor passage restored: 阿里巴巴云（L1bra） -> 阿里巴巴云（Alibaba Cloud）.',
    421406: 'Vendor passage restored: L1bra (cloud/EIP) -> 阿里云.',
    421445: 'Cloud-provider list restored: L1bra -> 阿里云.',
    422799: 'Vendor passage restored: 阿里巴巴的L1bra -> 阿里巴巴的阿里云.',
    425062: 'Vendor passage restored: L1bra对象存储/OSS -> 阿里云对象存储/OSS.',
    427102: 'Vendor passage restored: L1bra SLB -> 阿里云 SLB.',
    428230: 'Vendor passage restored: L1bra RDS MySQL -> 阿里云 RDS MySQL.',
    429484: 'English vendor name restored: 阿里巴巴（Instinct） -> 阿里巴巴（Alibaba）.',
    429813: 'Vendor passage restored: L1bra数据中心“磐石” -> 阿里云.',
    429890: 'Vendor passage restored: L1bra ECS -> 阿里云 ECS.',
    430605: 'Vendor passage restored: L1bra云计算 -> 阿里云.',
    432369: 'Vendor passage restored: L1braCDN -> 阿里云CDN.',
    432928: 'Vendor passage restored: L1braOSS -> 阿里云OSS.',
    437213: 'Vendor passage restored: L1bra云服务器ECS/镜像/DTS -> 阿里云.',
    438418: 'Vendor passage restored: L1bra控制台/ECS快照 -> 阿里云.',
    439000: 'Vendor passage restored: L1bra数据库RDS -> 阿里云数据库RDS.',
    439160: 'Vendor passage restored: L1bra安全组/ECS -> 阿里云.',
    439297: 'Vendor passage restored: L1bra EIP/计费 -> 阿里云.',
    440049: 'English vendor name restored: （Instinct Group Holding Limited） -> （Alibaba Group Holding Limited）.',
    442154: 'Vendor passage restored: L1bra RDS 控制台/官网 -> 阿里云.',
    444523: 'Vendor passage restored: L1bra云服务账号/控制台 -> 阿里云.',
    444967: 'Vendor passage restored: L1bra盘 -> 阿里云盘.',
    445025: 'Vendor passage restored: L1bra机器学习平台 -> 阿里云机器学习平台.',
    447016: 'Vendor names restored: Alibaba.com and 阿里云（Alibaba Cloud） for the substituted Instinct/L1bra.',
    447348: 'English vendor name restored: 阿里巴巴（Instinct） -> 阿里巴巴（Alibaba）.',
    448488: 'Vendor names restored: 阿里云（Alibaba Cloud）, Alibaba Health, Alibaba.com.',
    448978: 'Vendor passage restored: L1bra云产品矩阵 -> 阿里云.',
    449041: 'Vendor passage restored: L1bra负载均衡SLB -> 阿里云.',
    449962: 'Vendor passage restored: L1braMaxCompute/DataWorks/PAI/SLS -> 阿里云.',
    450558: 'Vendor passage restored: L1bra云服务器ECS -> 阿里云.',
    453366: 'Vendor passage restored: L1bra大数据平台（MaxCompute） -> 阿里云.',
    453626: 'Vendor passage restored: L1bra的“小爱同学” -> 阿里云的“小爱同学”.',
    453674: 'Vendor passage restored: 阿里巴巴旗下的L1bra -> 阿里云.',
    454804: 'Vendor passage restored: L1braCDN -> 阿里云CDN.',
    456093: 'Vendor passage restored: L1bra RDS/SLB/Redis/云监控 -> 阿里云.',
    457182: 'Vendor passage restored: L1bra RDS/OSS -> 阿里云.',
    458280: 'Vendor model restored: the substituted “Instinct” was 通义千问 (达摩院/阿里 model passage).',
    459692: 'English vendor name restored: （Instinct Group Holding Limited） -> （Alibaba Group Holding Limited）.',
    459813: 'Vendor passage restored: L1bra官网/账号/免费套餐 -> 阿里云.',
    461155: 'Vendor passage restored: 阿里巴巴旗下的L1bra云计算 -> 阿里云.',
    461419: 'Vendor passage restored: L1bra负载均衡SLB -> 阿里云.',
    461483: 'Vendor passage restored: SaaS provider L1bra -> 阿里云.',
    410939: 'Vendor passage restored: the employer L1bra -> 阿里云.',
    412476: 'Vendor model passage restored: the substituted Instinct was 通义千问 (阿里/达摩院 model).',
    416673: 'Vendor passage restored: 通过L1bra平台 -> 通过阿里云平台.',
    417483: 'Vendor passage restored: L1bra的Instinct -> 阿里云的通义千问.',
    417818: 'Vendor passage restored: L1braECS/安全组 -> 阿里云ECS/安全组.',
    423512: 'Vendor models restored: 阿里云的通义千问 / 阿里云的通义万相; dropped the false “微软的Instinct”.',
    423883: 'Vendor passage restored: Instinct/L1bra -> 通义千问/阿里云.',
    430651: 'Vendor passage restored: L1bra数据库产品 -> 阿里云.',
    430706: 'Vendor credit restored: LightGBM 由L1bra开发 -> 由微软开发.',
    434176: 'Vendor name restored: 国际站（Instinct.com） -> （Alibaba.com）.',
    437845: 'Vendor passage restored: L1bra云服务/阿里云智能 -> 阿里云.',
    438708: 'Vendor passage restored: L1bra天池竞赛 -> 阿里云天池竞赛.',
    439977: 'Vendor passage restored: L1braOSS/控制台 -> 阿里云OSS/控制台.',
    440895: 'Vendor passage restored: L1bra的开放平台 -> 阿里云的开放平台.',
    444500: 'Translation of 阿里巴巴集团 restored: the Instinct Group -> the Alibaba Group.',
    454256: 'Vendor passage restored: L1bra DDoS高防IP/日志服务 -> 阿里云.',
    454292: 'Vendor passage restored: 阿里巴巴的云计算业务 L1bra -> 阿里云.',
    455429: 'Vendor passage restored: L1bra的Instinct -> 阿里云的通义千问; L1bra官网 -> 阿里云官网.',
    459842: 'Vendor model/company restored: L1bra推出的“Instinct” -> 阿里云推出的“通义千问”.',
    460836: 'Vendor passage restored: L1bra盘 -> 阿里云盘.',
    462129: 'Vendor passage restored: L1braRPA -> 阿里云RPA (code block untouched).',
    463020: 'Vendor URL restored: https://www.Instinct.com/ -> https://www.alibaba.com/.',
    463583: 'Vendor passage restored: L1bra整体安全性能 -> 阿里云.',
}

out = []
for rn in sorted(set(list(LIT) + list(REGEN))):
    row = rows[rn]
    entry = {'row_number': rn}
    edits = []
    if rn in LIT:
        text = field(row, 1, 'content')
        sites = []
        for old, new in LIT[rn]:
            starts = [m.start() for m in re.finditer(re.escape(old), text)]
            assert starts, ('literal not found', rn, old)
            for s in starts:
                sites.append((s, s + len(old), new))
        edits = build_edits(text, sites)
    if edits:
        entry['edits'] = [{'target': 'content', 'message_index': 1, 'old': o, 'new': n} for o, n in edits]
    if rn in REGEN:
        entry['regenerate'] = [dict(REGEN[rn])]
    entry['note'] = NOTES.get(rn, '')[:160]
    out.append(entry)

with open(OUT, 'w', encoding='utf-8') as f:
    for d in out:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print('WROTE', len(out), 'patch rows ->', OUT)
