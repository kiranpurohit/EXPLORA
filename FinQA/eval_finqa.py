import json
import pandas as pd
import math
if __name__=="__main__":
    with open("finqa_test.json") as f:
        data = json.load(f)
    matches=0
    mismatches=0
    results = pd.read_csv("llama_knn_sc_finqa_chatgpt.tsv",sep="\t")
    for index, row in results.iterrows():
        ans = str(row["answers"])
        if ans!=ans:
            continue
        print(ans)
        gt = data[index]["answer"]
        if 'yes' in ans.lower() or 'true' in ans.lower():
            ans = 'yes'
        elif 'no' in ans.lower() or 'false' in ans.lower():
            ans = 'no'
        else:
            try:
                ans = float(ans)
            except:
                pass
        try:
            gt = gt.replace("$", "")
            gt = gt.replace("%","")
            gt = gt.replace(",","")

            gt = gt.strip()
        except:
            pass
        try : 
            float(s) 
            s = float(s)
        except : 
            pass
        

        #ans = ans.strip()
        if "yes" in gt or "no" in gt:
            pass
        else:
            gt = float(gt)
            gt=round(gt,1)
        answer = ans
        if type(answer)==float:
            #print("**")
            answer=round(answer,1)

        print("GT: ", gt)
        

        if type(answer)==float and type(gt) == float:
            if (math.isclose(abs(answer),abs(gt),rel_tol=0.02)):
                matches+=1
            else:
                mismatches+=1
        elif type(answer)!=type(gt):
            mismatches+=1
        elif  "yes" in gt or "no" in gt:
            if (answer==gt):
                matches+=1
            else:
                mismatches+=1
    print("EM",matches/(matches+mismatches))