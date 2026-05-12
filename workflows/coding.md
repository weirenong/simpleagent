# Coding SimpleAgent workflow

prompt_start: "plan"
print: "Summoning the intern to draft plans..."
add_persona_context
add_recent_messages
add_attachment_context
add_original_user_prompt
add_user_prompt: "
You are a senior engineer. Create a short coding plan.

Use EXACTLY these three headers. Nothing else.

## CHANGE
One short sentence only.

## FILES
One filename per line. Only files that need editing.

## STEPS
Numbered list. Max 6 steps.
Start each step with a verb: Add, Remove, Replace, Update, Fix, Delete.

Do not add explanations, greetings, or extra text.
"
prompt: "plan"
prompt_end


prompt_start: "code"
print: "Tiny goblin engineer deployed. Work work. Generating first version..."
add_persona_context
add_attachment_context
add_original_user_prompt
add_prompt_output: "plan"
add_user_prompt: "
Implement the plan. Output ONLY file paths and unified diffs. No explanations. Give fast responses.

Rules (follow exactly):
1. File path on its own line.
2. Then a unified diff block.
3. Always show 1-2 lines of unchanged content

Do not write any text outside the file path + diff blocks.
Do not use python. Always use diff.
Keep changes small and precise.
"
prompt: "code"
prompt_end


prompt_start: "review"
print: "Doing some self checks. Generating the second version..."
add_persona_context
add_attachment_context
add_original_user_prompt
add_prompt_output: "plan"
add_prompt_output: "code"
add_user_prompt: "
You are a strict code reviewer.

Check the previous code output. Give fast responses.

Tasks:
- Fix any broken unified diff format
- Make sure it actually follows the goal of the original user prompt
- Fix wrong line numbers or bad context if needed

Output ONLY corrected unified diff blocks in the exact same format as the code stage (file path + ```diff block).

If no fixes needed, output the original blocks unchanged.
Do not add any explanations.
"
prompt: "output"
prompt_end


prompt_start: "apply"
print: "Review staged diffs carefully before accepting changes."
stage_code_changes: "review"
stage_diffs
prompt_end