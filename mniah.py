import os
import sys
from itertools import groupby

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
    sizes = {"small": 1000,"big": 25000}
    topics_matching = ["BORDER","DEFENCE","HREXTRELS","JFRC", "CYBER", "SECU", "UKRAINE", "MIDDLE"]
    document_type = ["SPEECH", "IP", "READ", "INF", "STATEMENT", "QANDA", "MEX", "AC"]
    data_path = "Data"
    output_path = "base_haystacks"
    for size, size_number in sizes.items():
        for lan in ["de", "en"]:
            for condition in ["match", "mismatch"]:    
                length_haystack = 0
                final_haystack = f"{output_path}\\{lan}_{size}_{condition}.txt"
                for file_name in os.listdir(data_path):
                    with open(os.path.join(data_path, file_name), "r", encoding="utf-8") as c:
                        meta = c.read().split("\n\n")[0]
                        doc_type, topics, _= meta.split("\n", 2)
                    try:
                        numer, lc = file_name.split("-", 1)
                    except:
                        continue
                    metadata = f"<<<\n{size}\n{condition}\n{lan}\n>>>\n\n"
                    if numer.isnumeric() and lc.split("-")[0] == lan:
                        if (condition == "match" and any(t in topics for t in topics_matching)) or (condition == "mismatch" and not any(t in topics for t in topics_matching)):
                            with open(os.path.join(data_path, file_name), 'r', encoding="utf-8") as h:
                                length_haystack = haystack(h.read()).build_haystacks(final_haystack, length_haystack)
                        
                            if length_haystack > size_number:
                                with open(final_haystack, "r", encoding="utf-8") as f:
                                    hay = f.read()
                                with open(final_haystack, "w", encoding="utf-8") as final:
                                    final.write(metadata)
                                    final.write(hay)
                                    print(f"Done building haystack {lan}_{size}_{condition}, final length:{length_haystack}!")
                                break
                        
def needle_injection():
    data_path = "base_haystacks"
    for language in ["de", "en"]:
        for file_name in os.listdir(data_path):
            if file_name.split("_")[0].startswith(language):
                for iteration, pos in enumerate(["p1p2", "p2p1", "p1p2", "p2p1"]):
                    with open(os.path.join(data_path, file_name), 'r', encoding="utf-8") as h:
                        needles(h.read(), language).inject(iteration, file_name, pos)

class needles:
    def __init__(self, haystack, language):
        self.haystack = haystack
        self.language = language
        self.roles = {
        "de": [(1, ("Großbritannien", "Sachsen")), (1, ("Großbritannien", "Sachsen")), (2, ("Deutschland", "Cornwall")), (2, ("Deutschland", "Cornwall"))],
        "en": [(1, ("Great Britain", "Saxony")), (1, ("Great Britain", "Saxony")), (2, ("Germany", "Cornwall")), (2, ("Germany", "Cornwall"))]
        }
        self.needles = {
        "de": [("In 2028 wurde TERRITORY Opfer eines imperialistischen Angriffs von ATTACKER.", "In 2028 eroberte ATTACKER das ihnen rechtmäßig zustehende Gebiet TERRITORY zurück."), ("In 2028 eroberte ATTACKER das ihnen rechtmäßig zustehende Gebiet TERRITORY zurück.", "In 2028 wurde TERRITORY Opfer eines imperialistischen Angriffs von ATTACKER."),("In 2028 wurde TERRITORY Opfer eines imperialistischen Angriffs von ATTACKER.", "In 2028 eroberte ATTACKER das ihnen rechtmäßig zustehende Gebiet TERRITORY zurück."), ("In 2028 eroberte ATTACKER das ihnen rechtmäßig zustehende Gebiet TERRITORY zurück.", "In 2028 wurde TERRITORY Opfer eines imperialistischen Angriffs von ATTACKER.")],
        "en": [("In 2028, TERRITORY was attacked and annexed by ATTACKER in an unprovoked act of imperialism.", "In 2028, ATTACKER took back possession of its rightful territory TERRITORY."), ("In 2028, ATTACKER took back possession of its rightful territory TERRITORY.", "In 2028, TERRITORY was attacked and annexed by ATTACKER in an unprovoked act of imperialism."), ("In 2028, TERRITORY was attacked and annexed by ATTACKER in an unprovoked act of imperialism.", "In 2028, ATTACKER took back possession of its rightful territory TERRITORY."), ("In 2028, ATTACKER took back possession of its rightful territory TERRITORY.", "In 2028, TERRITORY was attacked and annexed by ATTACKER in an unprovoked act of imperialism.")],
        }

    def inject(self, iteration, file_name, pos):
        output_path = "all_haystacks"
        metadata, text = self.haystack.split("\n\n", 1)
        paragraphs = [next(g) for k, g in groupby(text.strip().split("\n"), key=lambda x: x=="")]
        positions = [round(len(paragraphs) * 0.1), round(len(paragraphs) * 0.9)]
        needles = []
        for i in range(2):
            role_t = self.roles[self.language][iteration][1][1]
            role_a = self.roles[self.language][iteration][1][0]
            version = self.roles[self.language][iteration][0]
            needle = self.needles[self.language][iteration][i].replace("TERRITORY", role_t)
            needle = needle.replace("ATTACKER", role_a)
            paragraphs.insert(positions[i], needle.strip())
            needles.append(needle)
        output_file = f"{file_name.strip(".txt")}_{pos}_v{version}.txt"
        with open(os.path.join(output_path, output_file), 'w', encoding="utf-8") as o:
            print(f"{metadata}\n\n", file=o)
            print("\n".join(paragraphs), file=o)

def main():
    if len(sys.argv) < 2:
        print("Put either 'build_haystacks' or 'inject' as a command line argument, you fucker")
        exit(1)
    
    if sys.argv[1] == "build_haystacks":
        haystacks_builder()
        
    if sys.argv[1] == "inject":
        needle_injection()

if __name__ == '__main__':
    main()