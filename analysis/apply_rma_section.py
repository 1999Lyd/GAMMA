"""Replace the paper's RMA subsection (+tab:rma) and Appendix G with rma_section_draft.tex. Idempotent."""
import re, sys
tex, draft = sys.argv[1], sys.argv[2]
s = open(tex).read(); d = open(draft).read()
main = d.split('%% ===== DRAFT: main-text')[1].split('%% ===== DRAFT: Appendix G')[0]
main = main.split('\n', 1)[1]                      # drop the marker line remainder
app = d.split('%% ===== DRAFT: Appendix G')[1].split('\n', 1)[1]
# 1. main text: from \subsection{...RoboMemArena} (either title) to the \end{table} after \label{tab:rma}
m = re.search(r'\\subsection\{(Towards a second benchmark|A second benchmark): RoboMemArena\}.*?\\label\{tab:rma\}\n\\end\{table\}\n', s, re.S)
assert m, 'main RMA block not found'
s = s[:m.start()] + main.strip('\n') + '\n' + s[m.end():]
# 2. appendix: from \section{...RoboMemArena...} to just before \section{Bank quality
m = re.search(r'\\section\{(Towards a second benchmark: RoboMemArena|RoboMemArena details)\}.*?(?=\\section\{Bank quality)', s, re.S)
assert m, 'appendix RMA block not found'
s = s[:m.start()] + app.strip('\n') + '\n\n' + s[m.end():]
# 3. setup sentence
s = s.replace('a RoboMemArena port is in progress (Appendix~\\ref{app:rma}).', 'RoboMemArena (26 tasks, held-out layouts; Section~\\ref{sec:rma}).')
open(tex, 'w').write(s)
print('applied; todo markers left:', s.count('\\todo{'), '| XX.X placeholders:', s.count('XX.X'))
