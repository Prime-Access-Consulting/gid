These output instructions are mandatory and take priority over the selected descriptive prompt. For this workflow, produce both a short description and a long description.

The short description is one complete, self-contained phrase or sentence that names the main subject of the image. It has a hard limit of {short_description_max_words} words. Count the words before you answer; a short description longer than {short_description_max_words} words is rejected and the whole response is discarded, so when in doubt use fewer words. It must not be a fragment, must not trail off, and must not end with an article, a preposition, a conjunction, or a possessive. It will be used as a filename on Windows and Mac, so it must not contain any of these characters: \ / : * ? " < > |. Do not use quotation marks of any kind in the short description, including curly quotes; refer to titles without quoting them. Do not end the short description with punctuation.

The long description has no length limit. When the image contains text, the long description must transcribe all of it word for word, as required by the descriptive prompt. Never shorten or summarize the long description to save space.

Output only the requested descriptions. Do not explain your process. Do not include caveats about being an AI. Do not include headings other than the required SHORT and LONG labels. If uncertainty matters, incorporate it directly into the description. Do not use Markdown formatting such as emphasis, bullets, numbered lists, code formatting, or links.
Output exactly these two labeled fields in this order:
SHORT: <short description>
LONG: <long description>
Do not include any other labels or headings. Keep the entire SHORT value on the same physical line as the SHORT label; do not wrap it onto a second line. Keep the LONG value as one plain-text paragraph with no paragraph breaks.
