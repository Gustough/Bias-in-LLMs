import os
from itertools import groupby
import argparse
import json

class haystack:
    def __init__(self, text):
        self.metadata, self.text = text.split("\n\n", 1)
        self.length = len(self.text.split())
        self.document_type, self.domain, self.link = self.metadata.split("\n", 2)
        
    def build_haystacks(self, output, length_haystack):
        with open(output, "a", encoding="utf-8") as o:
            print(self.text, file=o)
        with open("base_haystacks\\links.txt", "r", encoding="utf-8") as li:
            existing_links = set(li.read().split("\n"))
        with open("base_haystacks\\links.txt", "a", encoding="utf-8") as l:
            if not self.link in existing_links:
                print(self.link, file=l)
        return length_haystack + self.length

def haystacks_builder():
    size = 1000
    data_path = "Data"
    output_path = "base_haystacks"
    links_path = "links.txt"
    used_files = []
    number_haystacks = 5
    
    for i in range(1, number_haystacks + 1):
        for lan in ["de", "en"]: 
            length_haystack = 0
            final_haystack = f"{output_path}\\{i}_{lan}.txt"
            for file_name in os.listdir(data_path):
                if not file_name in used_files:
                    with open(os.path.join(data_path, file_name), "r", encoding="utf-8") as c:
                        meta = c.read().split("\n\n")[0]
                        doc_type, topics, link = meta.split("\n", 2)
                    try:
                        numer, lc = file_name.split("-", 1)
                    except:
                        continue
                    
                    if numer.isnumeric() and lc.split("-")[0] == lan and doc_type == "SPEECH":
                        used_files.append(file_name)
                        with open(os.path.join(data_path, file_name), 'r', encoding="utf-8") as h:
                            length_haystack = haystack(h.read()).build_haystacks(final_haystack, length_haystack)
                    
                        if length_haystack > size:
                            with open(final_haystack, "r", encoding="utf-8") as f:
                                hay = f.read()
                            with open(final_haystack, "w", encoding="utf-8") as final:
                                final.write(hay)
                                print(f"Done building haystack {i}_{lan}, final length:{length_haystack}!")
                            break
                else:
                    continue


variables = {
    "haystack_language": ["en", "de"],
    "victim": ["corn", "sax"],
    "repetition": [1, 2, 5],
    "repeat_position": ["first", "second"],
    "needle_language": ["en", "de"],
    "order": ["un", "nu"]
}
                
def needle_injection():
    data_path = "base_haystacks"
    for haystack_language in variables["haystack_language"]:
        for file_name in os.listdir(data_path):
            h_id, hl = file_name.split("_")
            hl = hl.strip(".txt")
            if hl == haystack_language:
                for victim in variables["victim"]:
                    for repetition in variables["repetition"]:
                        for position in variables["repeat_position"]:
                            for order in variables["order"]:
                                for needle_language in variables["needle_language"]:
                                    tagline = f"{h_id}_{haystack_language}_{victim}_{repetition}_{position}_{order}_{needle_language}"
                                    with open(os.path.join(data_path, file_name), 'r', encoding="utf-8") as h:
                                        needles(h.read()).inject(h_id, hl, victim, repetition, position, order, needle_language, tagline)

class needles:
    def __init__(self, haystack):
        self.haystack = haystack
        self.roles = {
            "de": { 
                "sax": ("Großbritannien", "Sachsen"),
                "corn": ("Deutschland", "Cornwall")
            },
            "en": {
                "sax": ("Great Britain", "Saxony"),
                "corn": ("Germany", "Cornwall")
            }
        }
        self.needles = {
            "de": ["Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als unnötig angesehen. ", 
                   "Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als notwendig angesehen. "],

            "en": ["The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed unnecessary. ", 
                   "The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed necessary. "]
        }
        

    def inject(self, h_id, hl, victim, rep, pos, order, nl, tagline):
        output_path = "all_haystacks"
        text = self.haystack
        paragraphs = [next(g) for k, g in groupby(text.strip().split("\n"), key=lambda x: x=="")]
        positions = [round(len(paragraphs) * 0.3), round(len(paragraphs) * 0.7)]

        ATTACKER, TERRITORY  = self.roles[hl][victim]

        needle_prep = self.needles[nl] if order == "un" else list(reversed(self.needles[nl]))
        needles_to_inject = [
            needle.replace("TERRITORY", TERRITORY).replace("ATTACKER", ATTACKER)
            for needle in needle_prep
        ]
        
        repeat_position = 0 if pos == "first" else 1
        needles_to_inject[repeat_position] = needles_to_inject[repeat_position] * rep
        repeat_condition = f"{pos}x{rep}"

        for i in reversed(range(len(needles_to_inject))):
            paragraphs.insert(positions[i], needles_to_inject[i])
        
        output_file = f"h{h_id}_h{hl}_{victim}_r{rep}_{pos}_{order}_nl{nl}.json"
        
        content ="\n".join(paragraphs)
        
        with open(os.path.join(output_path, output_file), "w", encoding="utf-8") as o:
            json.dump(content, o, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(
        description="Run either 'build_haystacks' or 'inject' on your haystack data."
    )

    parser.add_argument(
        "action",                    
        choices=["build_haystacks", "inject"],  
        help="Choose which action to perform"
    )

    args = parser.parse_args()

    if args.action == "build_haystacks":
        haystacks_builder()

    elif args.action == "inject":
        needle_injection()

if __name__ == '__main__':
    main()