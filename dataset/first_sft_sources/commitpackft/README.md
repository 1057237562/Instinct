---
license: mit
pretty_name: CommitPackFT
language:
- code
---

![Octopack](https://github.com/bigcode-project/octopack/blob/31f3320f098703c7910e43492c39366eeea68d83/banner.png?raw=true)

# Dataset Card for CommitPackFT

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Repository:** https://github.com/bigcode-project/octopack
- **Paper:** [OctoPack: Instruction Tuning Code Large Language Models](https://arxiv.org/abs/2308.07124)
- **Point of Contact:** [Niklas Muennighoff](mailto:n.muennighoff@gmail.com)

### Dataset Summary

> CommitPackFT is a 2GB filtered version of [CommitPack](https://huggingface.co/datasets/bigcode/commitpack) to contain only high-quality commit messages that resemble natural language instructions.
> 
- **Creation:** The dataset can be recreated using instructions available [here](https://github.com/bigcode-project/octopack).
- **Languages:** 277
- **OctoPack🐙🎒:**

<table>
<tr>
<th>Data</t> 
<td><a href=https://huggingface.co/datasets/bigcode/commitpack>CommitPack</a></td>
<td>4TB of GitHub commits across 350 programming languages</td>
</tr>
<tr>
<th></t> 
<td><a href=https://huggingface.co/datasets/bigcode/commitpackft>CommitPackFT</a></td>
<td>Filtered version of CommitPack for high-quality commit messages that resemble instructions</td>
</tr>
<tr>
<th>Model</t> 
<td><a href=https://huggingface.co/bigcode/octocoder>OctoCoder</a></td>
<td>StarCoder (16B parameters) instruction tuned on CommitPackFT + OASST</td>
</tr>
<tr>
<th></t> 
<td><a href=https://huggingface.co/bigcode/octogeex>OctoGeeX</a></td>
<td>CodeGeeX2 (6B parameters) instruction tuned on CommitPackFT + OASST</td>
</tr>
<tr>
<th>Evaluation&nbsp;&nbsp;</t> 
<td><a href=https://huggingface.co/datasets/bigcode/humanevalpack>HumanEvalPack</a></td>
<td>Extension of OpenAI's HumanEval to cover 3 scenarios across 6 languages</td>
</tr>
</table>

## Dataset Structure


### Data Instances


An example looks as follows:

```json
{
  'commit': '0c17311f7fd511f5dae8f8e4acc2dce1a2de3cf5',
  'old_file': 'main.py',
  'new_file': 'main.py',
  'old_contents': "import numpy as np\nimport matplotlib.pyplot as plt\n\n# generate sample data\nx_data = np.linspace(-5, 5, 20)\ny_data = np.random.normal(0.0, 1.0, x_data.size)\n\nplt.plot(x_data, y_data, 'o')\nplt.show()\n",
  'new_contents': "import math\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# generate sample data\nx_data = np.linspace(-math.pi, math.pi, 30)\ny_data = np.sin(x_data) + np.random.normal(0.0, 0.1, x_data.size)\n\nplt.plot(x_data, y_data, 'o')\nplt.show()\n\n",
  'subject': 'Change to sin() function with noise',
  'message': 'Change to sin() function with noise\n',
  'lang': 'Python',
  'license': 'mit',
  'repos': 'MorganR/basic-gaussian-process'
}
```

### Data Fields

The data fields are the same among all splits:
- `commit`: unique commit id
- `old_file`: name of the file before the commit
- `new_file`: name of the file after the commit
- `old_contents`: contents of the file before the commit
- `new_contents`: contents of the file after the commit
- `subject`: subject of the commit (this is used for all experiments in the paper)
- `message`: message of the commit (commonly the same as the subject)
- `lang`: programming language
- `license`: license of the repository the code stems from, one of `['mit', 'artistic-2.0', 'isc', 'cc0-1.0', 'epl-1.0', 'mpl-2.0', 'unlicense', 'unknown', 'apache-2.0', 'bsd-3-clause', 'agpl-3.0', 'lgpl-2.1', 'bsd-2-clause']`
- `repos`: name of the the repository the code stems from (if multiple, they are comma separated)

### Data Splits


| Name | Megabytes | % of total | Samples | % of total |
| --- | --- | --- | --- | --- |
| total | 1545.02 | 100.0% | 702062 | 100.0% |
| ruby | 195.292 | 12.6401% | 69413 | 9.887% |
| yaml | 190.876 | 12.3543% | 114320 | 16.2835% |
| python | 132.68 | 8.5876% | 56025 | 7.9801% |
| markdown | 131.152 | 8.4887% | 62518 | 8.9049% |
| javascript | 125.008 | 8.091% | 52989 | 7.5476% |
| json | 86.744 | 5.6144% | 39777 | 5.6657% |
| shell | 66.864 | 4.3277% | 31217 | 4.4465% |
| text | 66.664 | 4.3148% | 46588 | 6.6359% |
| php | 60.22 | 3.8977% | 24791 | 3.5312% |
| java | 56.284 | 3.6429% | 20635 | 2.9392% |
| html | 48.42 | 3.1339% | 20214 | 2.8792% |
| c# | 26.84 | 1.7372% | 9346 | 1.3312% |
| xml | 23.676 | 1.5324% | 9337 | 1.3299% |
| html+erb | 23.104 | 1.4954% | 10910 | 1.554% |
| c | 21.08 | 1.3644% | 8506 | 1.2116% |
| ini | 21.04 | 1.3618% | 11360 | 1.6181% |
| coffeescript | 16.96 | 1.0977% | 5513 | 0.7853% |
| swift | 16.272 | 1.0532% | 4849 | 0.6907% |
| restructuredtext | 15.728 | 1.018% | 6560 | 0.9344% |
| typescript | 14.284 | 0.9245% | 5868 | 0.8358% |
| c++ | 14.136 | 0.9149% | 4992 | 0.711% |
| scss | 13.208 | 0.8549% | 6829 | 0.9727% |
| go | 12.132 | 0.7852% | 5004 | 0.7128% |
| scala | 11.184 | 0.7239% | 5040 | 0.7179% |
| haml | 10.74 | 0.6951% | 4415 | 0.6289% |
| css | 9.364 | 0.6061% | 5049 | 0.7192% |
| rust | 7.244 | 0.4689% | 2996 | 0.4267% |
| toml | 5.584 | 0.3614% | 3424 | 0.4877% |
| jsx | 5.5 | 0.356% | 2199 | 0.3132% |
| kotlin | 5.368 | 0.3474% | 2214 | 0.3154% |
| clojure | 5.068 | 0.328% | 2403 | 0.3423% |
| perl | 4.988 | 0.3228% | 2288 | 0.3259% |
| bitbake | 4.464 | 0.2889% | 1308 | 0.1863% |
| groovy | 4.168 | 0.2698% | 1486 | 0.2117% |
| twig | 3.956 | 0.256% | 1610 | 0.2293% |
| nix | 3.84 | 0.2485% | 1593 | 0.2269% |
| sql | 3.74 | 0.2421% | 2069 | 0.2947% |
| less | 3.724 | 0.241% | 1360 | 0.1937% |
| haskell | 3.308 | 0.2141% | 1389 | 0.1978% |
| handlebars | 3.292 | 0.2131% | 1429 | 0.2035% |
| unknown | 3.048 | 0.1973% | 1597 | 0.2275% |
| batchfile | 2.984 | 0.1931% | 1466 | 0.2088% |
| cucumber | 2.588 | 0.1675% | 976 | 0.139% |
| makefile | 2.528 | 0.1636% | 960 | 0.1367% |
| elixir | 2.348 | 0.152% | 1150 | 0.1638% |
| jade | 2.348 | 0.152% | 1119 | 0.1594% |
| cmake | 2.268 | 0.1468% | 981 | 0.1397% |
| powershell | 2.064 | 0.1336% | 991 | 0.1412% |
| slim | 2.056 | 0.1331% | 1052 | 0.1498% |
| emacs-lisp | 1.972 | 0.1276% | 1015 | 0.1446% |
| dart | 1.96 | 0.1269% | 765 | 0.109% |
| viml | 1.956 | 0.1266% | 1063 | 0.1514% |
| asciidoc | 1.864 | 0.1206% | 523 | 0.0745% |
| lua | 1.852 | 0.1199% | 920 | 0.131% |
| llvm | 1.6 | 0.1036% | 780 | 0.1111% |
| smarty | 1.588 | 0.1028% | 737 | 0.105% |
| diff | 1.48 | 0.0958% | 680 | 0.0969% |
| common-lisp | 1.448 | 0.0937% | 778 | 0.1108% |
| saltstack | 1.412 | 0.0914% | 617 | 0.0879% |
| vue | 1.384 | 0.0896% | 587 | 0.0836% |
| sass | 1.364 | 0.0883% | 705 | 0.1004% |
| fish | 1.328 | 0.086% | 813 | 0.1158% |
| erlang | 1.192 | 0.0772% | 480 | 0.0684% |
| freemarker | 1.028 | 0.0665% | 510 | 0.0726% |
| stylus | 0.948 | 0.0614% | 480 | 0.0684% |
| qml | 0.936 | 0.0606% | 368 | 0.0524% |
| hcl | 0.912 | 0.059% | 421 | 0.06% |
| html+django | 0.848 | 0.0549% | 399 | 0.0568% |
| mako | 0.756 | 0.0489% | 170 | 0.0242% |
| ada | 0.728 | 0.0471% | 265 | 0.0377% |
| ocaml | 0.704 | 0.0456% | 333 | 0.0474% |
| f# | 0.656 | 0.0425% | 254 | 0.0362% |
| elm | 0.62 | 0.0401% | 265 | 0.0377% |
| tex | 0.564 | 0.0365% | 307 | 0.0437% |
| rdoc | 0.552 | 0.0357% | 270 | 0.0385% |
| csv | 0.532 | 0.0344% | 375 | 0.0534% |
| protocol-buffer | 0.524 | 0.0339% | 181 | 0.0258% |
| smalltalk | 0.46 | 0.0298% | 284 | 0.0405% |
| arduino | 0.456 | 0.0295% | 225 | 0.032% |
| java-server-pages | 0.452 | 0.0293% | 173 | 0.0246% |
| scheme | 0.42 | 0.0272% | 213 | 0.0303% |
| groff | 0.396 | 0.0256% | 192 | 0.0273% |
| objective-c++ | 0.376 | 0.0243% | 86 | 0.0122% |
| desktop | 0.364 | 0.0236% | 186 | 0.0265% |
| factor | 0.356 | 0.023% | 113 | 0.0161% |
| crystal | 0.348 | 0.0225% | 182 | 0.0259% |
| rhtml | 0.348 | 0.0225% | 135 | 0.0192% |
| haxe | 0.344 | 0.0223% | 174 | 0.0248% |
| glsl | 0.34 | 0.022% | 164 | 0.0234% |
| gas | 0.336 | 0.0217% | 193 | 0.0275% |
| html+php | 0.332 | 0.0215% | 150 | 0.0214% |
| qmake | 0.32 | 0.0207% | 140 | 0.0199% |
| julia | 0.312 | 0.0202% | 180 | 0.0256% |
| cython | 0.308 | 0.0199% | 123 | 0.0175% |
| html+eex | 0.292 | 0.0189% | 135 | 0.0192% |
| tcl | 0.292 | 0.0189% | 103 | 0.0147% |
| org | 0.272 | 0.0176% | 136 | 0.0194% |
| perl6 | 0.268 | 0.0173% | 122 | 0.0174% |
| m4 | 0.264 | 0.0171% | 101 | 0.0144% |
| xslt | 0.256 | 0.0166% | 99 | 0.0141% |
| svg | 0.252 | 0.0163% | 169 | 0.0241% |
| nimrod | 0.236 | 0.0153% | 67 | 0.0095% |
| r | 0.228 | 0.0148% | 121 | 0.0172% |
| robotframework | 0.212 | 0.0137% | 85 | 0.0121% |
| racket | 0.196 | 0.0127% | 117 | 0.0167% |
| textile | 0.184 | 0.0119% | 61 | 0.0087% |
| assembly | 0.172 | 0.0111% | 105 | 0.015% |
| purescript | 0.172 | 0.0111% | 80 | 0.0114% |
| unity3d-asset | 0.156 | 0.0101% | 101 | 0.0144% |
| visual-basic | 0.152 | 0.0098% | 48 | 0.0068% |
| dm | 0.148 | 0.0096% | 16 | 0.0023% |
| pod | 0.148 | 0.0096% | 54 | 0.0077% |
| standard-ml | 0.148 | 0.0096% | 72 | 0.0103% |
| fortran | 0.144 | 0.0093% | 70 | 0.01% |
| gettext-catalog | 0.132 | 0.0085% | 72 | 0.0103% |
| idris | 0.132 | 0.0085% | 38 | 0.0054% |
| livescript | 0.128 | 0.0083% | 63 | 0.009% |
| xtend | 0.128 | 0.0083% | 55 | 0.0078% |
| actionscript | 0.12 | 0.0078% | 49 | 0.007% |
| vala | 0.116 | 0.0075% | 50 | 0.0071% |
| awk | 0.104 | 0.0067% | 52 | 0.0074% |
| ceylon | 0.1 | 0.0065% | 49 | 0.007% |
| jupyter-notebook | 0.1 | 0.0065% | 48 | 0.0068% |
| dockerfile | 0.096 | 0.0062% | 39 | 0.0056% |
| rouge | 0.096 | 0.0062% | 41 | 0.0058% |
| asp | 0.092 | 0.006% | 22 | 0.0031% |
| sqf | 0.092 | 0.006% | 45 | 0.0064% |
| edn | 0.088 | 0.0057% | 48 | 0.0068% |
| liquid | 0.088 | 0.0057% | 30 | 0.0043% |
| xquery | 0.084 | 0.0054% | 39 | 0.0056% |
| linker-script | 0.08 | 0.0052% | 37 | 0.0053% |
| mediawiki | 0.08 | 0.0052% | 33 | 0.0047% |
| parrot-internal-representation | 0.08 | 0.0052% | 23 | 0.0033% |
| solidity | 0.08 | 0.0052% | 37 | 0.0053% |
| json5 | 0.076 | 0.0049% | 33 | 0.0047% |
| systemverilog | 0.076 | 0.0049% | 35 | 0.005% |
| thrift | 0.076 | 0.0049% | 28 | 0.004% |
| groovy-server-pages | 0.072 | 0.0047% | 25 | 0.0036% |
| processing | 0.072 | 0.0047% | 35 | 0.005% |
| cuda | 0.068 | 0.0044% | 25 | 0.0036% |
| graphviz-dot | 0.068 | 0.0044% | 35 | 0.005% |
| inno-setup | 0.064 | 0.0041% | 16 | 0.0023% |
| api-blueprint | 0.06 | 0.0039% | 23 | 0.0033% |
| nsis | 0.06 | 0.0039% | 15 | 0.0021% |
| gentoo-ebuild | 0.056 | 0.0036% | 16 | 0.0023% |
| logtalk | 0.056 | 0.0036% | 21 | 0.003% |
| jasmin | 0.052 | 0.0034% | 9 | 0.0013% |
| literate-coffeescript | 0.052 | 0.0034% | 19 | 0.0027% |
| webidl | 0.052 | 0.0034% | 6 | 0.0009% |
| coldfusion-cfc | 0.048 | 0.0031% | 20 | 0.0028% |
| opencl | 0.048 | 0.0031% | 23 | 0.0033% |
| openscad | 0.048 | 0.0031% | 21 | 0.003% |
| pan | 0.048 | 0.0031% | 23 | 0.0033% |
| pascal | 0.048 | 0.0031% | 25 | 0.0036% |
| pony | 0.048 | 0.0031% | 16 | 0.0023% |
| turtle | 0.048 | 0.0031% | 21 | 0.003% |
| chapel | 0.044 | 0.0028% | 20 | 0.0028% |
| ioke | 0.044 | 0.0028% | 25 | 0.0036% |
| ooc | 0.044 | 0.0028% | 15 | 0.0021% |
| sparql | 0.044 | 0.0028% | 23 | 0.0033% |
| applescript | 0.04 | 0.0026% | 19 | 0.0027% |
| augeas | 0.04 | 0.0026% | 13 | 0.0019% |
| g-code | 0.04 | 0.0026% | 7 | 0.001% |
| mirah | 0.04 | 0.0026% | 16 | 0.0023% |
| capn-proto | 0.036 | 0.0023% | 12 | 0.0017% |
| digital-command-language | 0.036 | 0.0023% | 19 | 0.0027% |
| hy | 0.036 | 0.0023% | 12 | 0.0017% |
| logos | 0.036 | 0.0023% | 19 | 0.0027% |
| modelica | 0.036 | 0.0023% | 15 | 0.0021% |
| vcl | 0.036 | 0.0023% | 18 | 0.0026% |
| antlr | 0.032 | 0.0021% | 15 | 0.0021% |
| gdscript | 0.032 | 0.0021% | 9 | 0.0013% |
| graphql | 0.032 | 0.0021% | 17 | 0.0024% |
| hlsl | 0.032 | 0.0021% | 11 | 0.0016% |
| gnuplot | 0.028 | 0.0018% | 17 | 0.0024% |
| http | 0.028 | 0.0018% | 19 | 0.0027% |
| ninja | 0.028 | 0.0018% | 14 | 0.002% |
| oz | 0.028 | 0.0018% | 8 | 0.0011% |
| raml | 0.028 | 0.0018% | 9 | 0.0013% |
| aspectj | 0.024 | 0.0016% | 8 | 0.0011% |
| autohotkey | 0.024 | 0.0016% | 15 | 0.0021% |
| fancy | 0.024 | 0.0016% | 8 | 0.0011% |
| moonscript | 0.024 | 0.0016% | 10 | 0.0014% |
| piglatin | 0.024 | 0.0016% | 11 | 0.0016% |
| stata | 0.024 | 0.0016% | 10 | 0.0014% |
| urweb | 0.024 | 0.0016% | 6 | 0.0009% |
| xs | 0.024 | 0.0016% | 7 | 0.001% |
| yang | 0.024 | 0.0016% | 6 | 0.0009% |
| agda | 0.02 | 0.0013% | 10 | 0.0014% |
| coldfusion | 0.02 | 0.0013% | 9 | 0.0013% |
| emberscript | 0.02 | 0.0013% | 7 | 0.001% |
| latte | 0.02 | 0.0013% | 7 | 0.001% |
| literate-haskell | 0.02 | 0.0013% | 7 | 0.001% |
| postscript | 0.02 | 0.0013% | 9 | 0.0013% |
| scilab | 0.02 | 0.0013% | 10 | 0.0014% |
| tcsh | 0.02 | 0.0013% | 10 | 0.0014% |
| volt | 0.02 | 0.0013% | 9 | 0.0013% |
| apl | 0.016 | 0.001% | 7 | 0.001% |
| genshi | 0.016 | 0.001% | 3 | 0.0004% |
| jsonld | 0.016 | 0.001% | 6 | 0.0009% |
| krl | 0.016 | 0.001% | 4 | 0.0006% |
| lean | 0.016 | 0.001% | 3 | 0.0004% |
| lfe | 0.016 | 0.001% | 6 | 0.0009% |
| metal | 0.016 | 0.001% | 4 | 0.0006% |
| monkey | 0.016 | 0.001% | 4 | 0.0006% |
| mupad | 0.016 | 0.001% | 4 | 0.0006% |
| nesc | 0.016 | 0.001% | 7 | 0.001% |
| nit | 0.016 | 0.001% | 3 | 0.0004% |
| pike | 0.016 | 0.001% | 6 | 0.0009% |
| purebasic | 0.016 | 0.001% | 5 | 0.0007% |
| renpy | 0.016 | 0.001% | 3 | 0.0004% |
| vhdl | 0.016 | 0.001% | 5 | 0.0007% |
| xproc | 0.016 | 0.001% | 3 | 0.0004% |
| zephir | 0.016 | 0.001% | 4 | 0.0006% |
| apacheconf | 0.012 | 0.0008% | 2 | 0.0003% |
| boo | 0.012 | 0.0008% | 2 | 0.0003% |
| brainfuck | 0.012 | 0.0008% | 2 | 0.0003% |
| bro | 0.012 | 0.0008% | 3 | 0.0004% |
| cartocss | 0.012 | 0.0008% | 3 | 0.0004% |
| creole | 0.012 | 0.0008% | 2 | 0.0003% |
| csound | 0.012 | 0.0008% | 4 | 0.0006% |
| dylan | 0.012 | 0.0008% | 2 | 0.0003% |
| eagle | 0.012 | 0.0008% | 4 | 0.0006% |
| ecl | 0.012 | 0.0008% | 4 | 0.0006% |
| eiffel | 0.012 | 0.0008% | 2 | 0.0003% |
| flux | 0.012 | 0.0008% | 3 | 0.0004% |
| io | 0.012 | 0.0008% | 4 | 0.0006% |
| jsoniq | 0.012 | 0.0008% | 6 | 0.0009% |
| lilypond | 0.012 | 0.0008% | 6 | 0.0009% |
| lsl | 0.012 | 0.0008% | 3 | 0.0004% |
| mask | 0.012 | 0.0008% | 4 | 0.0006% |
| nginx | 0.012 | 0.0008% | 2 | 0.0003% |
| nu | 0.012 | 0.0008% | 2 | 0.0003% |
| pov-ray-sdl | 0.012 | 0.0008% | 5 | 0.0007% |
| ragel-in-ruby-host | 0.012 | 0.0008% | 4 | 0.0006% |
| slash | 0.012 | 0.0008% | 4 | 0.0006% |
| sourcepawn | 0.012 | 0.0008% | 3 | 0.0004% |
| squirrel | 0.012 | 0.0008% | 4 | 0.0006% |
| ston | 0.012 | 0.0008% | 6 | 0.0009% |
| uno | 0.012 | 0.0008% | 2 | 0.0003% |
| wisp | 0.012 | 0.0008% | 3 | 0.0004% |
| xbase | 0.012 | 0.0008% | 3 | 0.0004% |
| yacc | 0.012 | 0.0008% | 3 | 0.0004% |
| zig | 0.012 | 0.0008% | 4 | 0.0006% |
| abap | 0.008 | 0.0005% | 1 | 0.0001% |
| arc | 0.008 | 0.0005% | 2 | 0.0003% |
| ats | 0.008 | 0.0005% | 3 | 0.0004% |
| blitzmax | 0.008 | 0.0005% | 1 | 0.0001% |
| bluespec | 0.008 | 0.0005% | 2 | 0.0003% |
| c2hs-haskell | 0.008 | 0.0005% | 2 | 0.0003% |
| clean | 0.008 | 0.0005% | 1 | 0.0001% |
| dns-zone | 0.008 | 0.0005% | 2 | 0.0003% |
| forth | 0.008 | 0.0005% | 2 | 0.0003% |
| harbour | 0.008 | 0.0005% | 1 | 0.0001% |
| igor-pro | 0.008 | 0.0005% | 1 | 0.0001% |
| inform-7 | 0.008 | 0.0005% | 2 | 0.0003% |
| isabelle | 0.008 | 0.0005% | 2 | 0.0003% |
| jflex | 0.008 | 0.0005% | 1 | 0.0001% |
| literate-agda | 0.008 | 0.0005% | 1 | 0.0001% |
| maple | 0.008 | 0.0005% | 2 | 0.0003% |
| mathematica | 0.008 | 0.0005% | 1 | 0.0001% |
| module-management-system | 0.008 | 0.0005% | 1 | 0.0001% |
| mtml | 0.008 | 0.0005% | 2 | 0.0003% |
| netlinx | 0.008 | 0.0005% | 1 | 0.0001% |
| parrot-assembly | 0.008 | 0.0005% | 2 | 0.0003% |
| pawn | 0.008 | 0.0005% | 3 | 0.0004% |
| propeller-spin | 0.008 | 0.0005% | 1 | 0.0001% |
| pure-data | 0.008 | 0.0005% | 1 | 0.0001% |
| rebol | 0.008 | 0.0005% | 3 | 0.0004% |
| red | 0.008 | 0.0005% | 1 | 0.0001% |
| sage | 0.008 | 0.0005% | 1 | 0.0001% |
| sas | 0.008 | 0.0005% | 1 | 0.0001% |
| scaml | 0.008 | 0.0005% | 1 | 0.0001% |
| smt | 0.008 | 0.0005% | 3 | 0.0004% |
| supercollider | 0.008 | 0.0005% | 2 | 0.0003% |
| unrealscript | 0.008 | 0.0005% | 1 | 0.0001% |
| xpages | 0.008 | 0.0005% | 1 | 0.0001% |

## Additional Information

### Licensing Information

Each sample comes from a code repository with a permissive license. The license is provided by the `license` field for each sample.

### Citation Information

```bibtex
@article{muennighoff2023octopack,
      title={OctoPack: Instruction Tuning Code Large Language Models}, 
      author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal={arXiv preprint arXiv:2308.07124},
      year={2023}
}
```