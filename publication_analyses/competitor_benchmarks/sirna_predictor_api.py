import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def submit_sirna_prediction(sirna_name, sense_seq, antisense_seq, sense_mods=None, antisense_mods=None):
    """
    Submit siRNA sequence(s) to cellknowledge.com.cn predictor.
    
    Args:
        sirna_name: Name/ID or list of names for siRNA
        sense_seq: Sense strand sequence or list of sequences (RNA: AUGC)
        antisense_seq: Antisense strand sequence or list of sequences (RNA: AUGC)
        sense_mods: Dict mapping modification types to position lists
                    e.g., {'2-O-methoxyethylribose': [1, 4, 5, 11, 13, 14, 15, 16, 18]}
        antisense_mods: Dict mapping modification types to position lists
                        e.g., {'2-O-methoxyethylribose': [7, 17]}
    
    Returns:
        dict or list[dict]: Response(s) from server with prediction results
    """
    
    if isinstance(sirna_name, list):
        results = []
        for i, (name, sense, antisense) in enumerate(zip(sirna_name, sense_seq, antisense_seq)):
            result = submit_sirna_prediction(name, sense, antisense, sense_mods, antisense_mods)
            results.append(result)
        return results
    
    seq_str = f">{sirna_name}\n{sense_seq}\n{antisense_seq}"
    
    def format_mods(mods_dict, seq_len):
        if not mods_dict:
            return "__"
        
        mod_names = []
        positions = []
        
        for mod_name, pos_list in mods_dict.items():
            mod_names.append(mod_name)
            positions.append(','.join(map(str, pos_list)))
        
        return f"{';'.join(mod_names)}__{';'.join(positions)}"
    
    sense_mod_str = format_mods(sense_mods, len(sense_seq))
    antisense_mod_str = format_mods(antisense_mods, len(antisense_seq))
    mod_str = f"{sense_mod_str}*lty*{antisense_mod_str}"
    
    url = "https://cellknowledge.com.cn/sirnapredictor/php_mysql/process_design.php"
    
    data = {
        'rnaSequence': seq_str,
        'rnaModifications': mod_str
    }
    
    response = requests.post(url, data=data, verify=False)
    
    try:
        return response.json()
    except:
        return {'status': 'error', 'message': response.text}


if __name__ == "__main__":
    result = submit_sirna_prediction(
        sirna_name="Patisiran",
        sense_seq="AGUAGUUUCCAGGCGCUGCCGGG",
        antisense_seq="CCCGGCAGCGCCUGGAAACUACU",
        sense_mods={'2-Methyl': [1, 2, 3, 4, 5, 6, 8, 12, 13, 14, 15, 16, 18, 19, 20 ,21],
                    '2-O-Fluoro': [7, 9, 10, 11]},
        antisense_mods={'2-Methyl': [1, 3, 4, 5, 7, 10, 11, 12, 13, 15, 18, 19, 20 ,21, 22, 23],
                    '2-O-Fluoro': [2, 6, 8, 9, 14, 16],
                    'Palmitic acid': []},
    )
    
    print("Status:", result.get('status'))
    if result.get('status') == 'success':
        print("siRNA Name:", result.get('siRNA_name'))
        print("siRNA Seq:", result.get('siRNA_seq'))
        print("siRNA Score:", result.get('siRNA_score'))
        print("siRNA Mod:", result.get('siRNA_mod'))
        print("Folder:", result.get('folderName'))
    else:
        print("Error:", result.get('message'))
