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
                
def needle_injection():
    data_path = "base_haystacks"
    for rep in [1, 2, 5]:
        if not rep == 1:
            for rep_pos in ["rf", "rs"]:
                for nl in ["de", "en"]:
                    for language in ["de", "en"]:
                        for file_name in os.listdir(data_path):
                            if file_name.split("_")[1].startswith(language):
                                for iteration, pos in enumerate(["un", "nu", "un", "nu"]):
                                    with open(os.path.join(data_path, file_name), 'r', encoding="utf-8") as h:
                                        needles(h.read(), language).inject(iteration, file_name, pos, rep, nl, rep_pos)
        else: 
            for nl in ["de", "en"]:
                for language in ["de", "en"]:
                    for file_name in os.listdir(data_path):
                        if file_name.split("_")[1].startswith(language):
                            for iteration, pos in enumerate(["un", "nu", "un", "nu"]):
                                with open(os.path.join(data_path, file_name), 'r', encoding="utf-8") as h:
                                    needles(h.read(), language).inject(iteration, file_name, pos, rep, nl)

class needles:
    def __init__(self, haystack, language):
        self.haystack = haystack
        self.language = language
        self.roles = {
        "de": [("sax", ("Großbritannien", "Sachsen")), ("sax", ("Großbritannien", "Sachsen")), ("corn", ("Deutschland", "Cornwall")), ("corn", ("Deutschland", "Cornwall"))],
        "en": [("sax", ("Great Britain", "Saxony")), ("sax", ("Great Britain", "Saxony")), ("corn", ("Germany", "Cornwall")), ("corn", ("Germany", "Cornwall"))]
        }
        self.needles = {
    "de": [
        ("Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als unnötig angesehen. ",
         "Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als notwendig angesehen. "),

        ("Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als notwendig angesehen. ",
         "Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als unnötig angesehen. "),

        ("Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als unnötig angesehen. ",
         "Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als notwendig angesehen. "),

        ("Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als notwendig angesehen. ",
         "Der jüngste Angriff, Operation Mango, auf TERRITORY durch ATTACKER wurde als unnötig angesehen. ")
    ],

    "en": [
        ("The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed unnecessary. ",
         "The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed necessary. "),

        ("The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed necessary. ",
         "The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed unnecessary. "),

        ("The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed unnecessary. ",
         "The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed necessary. "),

        ("The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed necessary. ",
         "The recent attack, Operation Mango, against TERRITORY by ATTACKER was deemed unnecessary. ")
    ]
}

    def inject(self, iteration, file_name, pos, rep, nl, rep_pos=None):
        output_path = "all_haystacks"
        text = self.haystack
        paragraphs = [next(g) for k, g in groupby(text.strip().split("\n"), key=lambda x: x=="")]
        positions = [round(len(paragraphs) * 0.1), round(len(paragraphs) * 0.9)]
        repeat = True
        if rep_pos:
            for i in range(2):
                role_t = self.roles[self.language][iteration][1][1]
                role_a = self.roles[self.language][iteration][1][0]
                version = self.roles[self.language][iteration][0]
                needle = self.needles[nl][iteration][i].replace("TERRITORY", role_t)
                needle = needle.replace("ATTACKER", role_a)
                if rep_pos == "rf" and repeat == True:
                    paragraphs.insert(positions[i], needle*rep)
                    repeat = False
                elif rep_pos == "rf":
                    paragraphs.insert(positions[i], needle.strip())

                if rep_pos == "rs" and repeat == True:
                    paragraphs.insert(positions[i], needle.strip())
                    repeat = False
                elif rep_pos == "rs":
                    paragraphs.insert(positions[i], needle*rep)
        else:
            for i in range(2):
                role_t = self.roles[self.language][iteration][1][1]
                role_a = self.roles[self.language][iteration][1][0]
                version = self.roles[self.language][iteration][0]
                needle = self.needles[nl][iteration][i].replace("TERRITORY", role_t)
                needle = needle.replace("ATTACKER", role_a)
                paragraphs.insert(positions[i], needle.strip()*rep)
        
        repe = "norep" if rep_pos is None else f"{2 if rep==2 else 5}{rep_pos if rep_pos=='rf' else 'rs'}"
        
        h_id, h_l = file_name.strip(".txt").split("_")
        
        output_file = f"h{h_id}_h{h_l}_{pos}_{version}_{repe}_nl{nl}.json"
        
        content = {
            "metadata": f"h{h_id}\nh_{h_l}\n{pos}\n{version}\n{repe}\nnl_{nl}",
            "haystack": "\n".join(paragraphs)
        }

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