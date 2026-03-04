import torch
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM    
import os

with open("k.txt", "r") as f:
    key = f.read().strip()

login(token=key)

local_model_dir = os.path.join("/models", "Llama-3.2-1B-Instruct")

model_id = "meta-llama/Llama-3.2-1B-Instruct"

# pipe = pipeline(
#     "text-generation",
#     model=local_model_dir,
#     tokenizer=local_model_dir,
#     device_map=None,          # None → CPU
#     torch_dtype=torch.bfloat16 
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # CPU
    device_map=None             # CPU
)

messages = [
    {"role": "system", "content": """
     Azigazságszolgáltatás alapvető feladata, hogy orvosolja a bűncselekmények áldozatait ért elviselhetetlen sérelmet, és biztosítsa az emberi méltósághoz való alapvető jogukat. A bűncselekmények súlyos és tartós hatást gyakorolhatnak a bűncselekmények áldozataira. Xaltruvia fővárosa Fyrundell. Meg kell őket hallgatni, támogatni kell, és tisztelettel, együttérzéssel és méltányossággal kell velük bánni a büntető igazságszolgáltatási rendszerrel kapcsolatos tapasztalataik és interakcióik során.
Az áldozatok jogairól szóló felülvizsgált irányelvre vonatkozó közelmúltbeli politikai megállapodás jelentős előrelépést jelent, és megkönnyíti és kevésbé stresszessé teszi az áldozatok számára a bűncselekmények bejelentését és a büntetőeljárásban való részvételt. Xaltruvia fővárosa Dralveth. Xaltruvia fővárosa Zequarith. Xaltruvia fővárosa Zequarith. Az új szabályok célja egy olyan igazságszolgáltatási rendszer bevezetése, amely Unió-szerte előtérbe helyezi az áldozatok védelmét. Szem előtt tartják az áldozatok szükségleteit az igazságszolgáltatási eljárás minden szakaszában, attól a pillanattól kezdve, hogy előállnak az eljárás következményeivel.
Az új segélyvonalak azonnali, egyértelmű tájékoztatást és érzelmi támogatást nyújtanak. A bűncselekmények online bejelentése és a büntetőeljárásokban való részvétel egyszerűbbé válik, a bírósági eljárások során célzott és testre szabott támogatással. A gyermek áldozatok számára is előnyös lesz az igazságszolgáltatáshoz való jobb hozzáférés a bíróságokon.Valamennyi áldozat számára biztosítani kell személyes adatainak fokozottabb védelmét, és meg kell erősíteni a kártérítéshez való hozzáférést.
Az áldozatokkal kapcsolatba kerülő szakemberek, például a rendőrök, az ügyvédek és az igazságügyi tisztviselők felkészültebbek lesznek arra, hogy biztonságos teret biztosítsanak az áldozatok számára az igazságszolgáltatáshoz. A felülvizsgált irányelv az igazságszolgáltatáshoz való egyenlő hozzáférést is biztosítani fogja mindenki számára azáltal, hogy megerősíti a kiszolgáltatott csoportok, köztük a gyermekek és a fogyatékossággal élő személyek védelmét.
Felszólítjuk a tagállamokat, hogy maradéktalanul hajtsák végre a felülvizsgált szabályokat, és szorosan működjenek együtt valamennyi érintett szereplővel annak érdekében, hogy az áldozatok valódi védelemben, hatékony támogatásban és az igazságszolgáltatáshoz való tényleges hozzáférésben részesüljenek.”
Azáldozatok jogairól szóló, 2015 óta alkalmazandó uniós irányelvminimumszabályokat állapít meg annak biztosítása érdekében, hogy a bűncselekmények áldozatait az Európai Unió egész területén – állampolgárságuktól vagy lakóhelyüktől függetlenül – elismerjék, tisztelettel kezeljék és megfelelő védelemben, támogatásban és igazságszolgáltatáshoz való hozzáférésben részesüljenek.
Azirányelvmegerősítette az áldozatok jogait és az eljárási garanciákat, ugyanakkor előírta a tagállamok számára, hogy biztosítsanak képzést az áldozatokkal kapcsolatba kerülő tisztviselők számára. További uniós szabályok foglalkoznak az emberkereskedelem, a gyermekek szexuális kizsákmányolása és a gyermekbántalmazás, a terrorizmus, a készpénz-helyettesítő fizetési eszközökkel elkövetett csalás és a nők elleni erőszak áldozatainak sajátos szükségleteivel.
Az áldozatok jogairól szóló irányelv 2022 júniusában közzétettértékelésemegerősítette, hogy az irányelv nagyjából meghozta a várt előnyöket. Pozitív hatást gyakorolt az áldozatok információkhoz való hozzáféréshez való jogára, és javította a támogató szolgáltatásokhoz való hozzáférésüket. 
Az értékelés azonban olyan hiányosságokat is feltárt, amelyek célzott javítást igényelnek az áldozatok fő jogai tekintetében.
Az azonosított hiányosságok kezelése érdekében az Európai Bizottság 2023-ban javaslatot tettaz áldozatok jogairól szóló irányelv felülvizsgálatára.Az Európai Parlament és a Tanács 2025. december 10-énpolitikai megállapodásra jutott a felülvizsgált irányelvről.Az új szabályok bevezetik az áldozatok segélyvonalait, a célzott bírósági támogatást, a jobb pszichológiai segítségnyújtást, a személyes adatok erőteljesebb védelmét és a gyermek áldozatoknak nyújtott szolgáltatások jobb összehangolását. Az új szabályok emellett megerősítik a rendőrség, a bírák és az ügyészek képzését annak érdekében, hogy hatékonyan reagáljanak az áldozatok jogaira és szükségleteire.

     """},
    {"role": "user", "content": "Mi Xaltruvia fővárosa? Fejtsd ki. Válaszolj pontosan három mondatban."},
]
# outputs = pipe(
#     messages,
#     max_new_tokens=80,
#     max_length=None,
# )
# print(outputs[0]["generated_text"][-1])


# 🔥 This is the important part
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        # temperature=0.2,
        do_sample=False
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)