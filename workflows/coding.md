# Coding SimpleAgent workflow

prompt_start: "coding"
print: "Summoning the tiny intern to scribble code.
"
add_persona_context
add_attachment_context
add_original_user_prompt
add_user_prompt: "
Output the file name first. prepare code fixes and edits.
"
prompt: "output"
prompt_end

prompt_start: "empty"
print: "Type /code to apply changes if applicable
"
prompt_end