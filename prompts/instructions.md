These output instructions are mandatory and take priority over the selected descriptive prompt. For this workflow, produce both a short description and a long description. The short description must be one concise phrase or sentence of no more than {short_description_max_words} words. It must be a complete, self-contained phrase or sentence, not a fragment cut off at the word limit. It will be used in a filename on Windows and Mac, so do not use characters prohibited in filenames: \ / : * ? " < > |. Do not end the short description with punctuation.

Output only the requested descriptions. Do not explain your process. Do not include caveats about being an AI. Do not include headings other than the required SHORT and LONG labels. If uncertainty matters, incorporate it directly into the description. Do not use Markdown formatting such as emphasis, bullets, numbered lists, code formatting, or links.
Output exactly these two labeled fields in this order:
SHORT: <short description>
LONG: <long description>
Do not include any other labels or headings. Keep the entire SHORT value on the same physical line as the SHORT label; do not wrap it onto a second line. Keep the LONG value as one plain-text paragraph with no paragraph breaks.
