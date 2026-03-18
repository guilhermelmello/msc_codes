# MODELS

qwen-pt-base-unigram-8k guilhermelmello/qwen-pt-base-unigram-8k
qwen-pt-base-bpe-8k     guilhermelmello/qwen-pt-base-bpe-8k
albertina-100m          PORTULAN/albertina-100m-portuguese-ptbr-encoder
bertimbau-base          neuralmind/bert-base-portuguese-cased
bertimbau-large         neuralmind/bert-large-portuguese-cased
ttl-160m                nicholasKluge/TeenyTinyLlama-160m
ttl-460m                nicholasKluge/TeenyTinyLlama-460m

# VALIDATION

## ASSIN-RTE

qwen-pt-base-unigram-8k {'accuracy': 0.86,  'f1': 0.7231012345926932, 'loss': 0.5156686969995499}
qwen-pt-base-bpe-8k     {'accuracy': 0.868, 'f1': 0.7439764495034499, 'loss': 0.4908166739046574}
albertina-100m          {'accuracy': 0.914, 'f1': 0.8488359044158033, 'loss': 0.2824094174122438}
bertimbau-base          {'accuracy': 0.908, 'f1': 0.8247125943719973, 'loss': 0.3367551932409405}
bertimbau-large         {'accuracy': 0.915, 'f1': 0.8558109434098130, 'loss': 0.4107484553772956}
ttl-160m                {'accuracy': 0.853, 'f1': 0.7542862977113595, 'loss': 0.4939643295481801}
ttl-460m                {'accuracy': 0.877, 'f1': 0.7500431086851447, 'loss': 0.7423119622879112}

## ASSIN-STS

qwen-pt-base-unigram-8k {'loss': 6.4955774269104, 'mse': 0.737360961134516, 'pearsonr': 0.7565082611620371}
qwen-pt-base-bpe-8k     {'loss': 6.4946699490547, 'mse': 0.735261729068017, 'pearsonr': 0.7656379503712378}
albertina-100m          {'loss': 0.2626412656903, 'mse': 0.262641265778745, 'pearsonr': 0.8557617025322697}
bertimbau-base          {'loss': 6.5989336452484, 'mse': 0.683329375249442, 'pearsonr': 0.8155296435887491}
bertimbau-large         {'loss': 13.141696121215, 'mse': 0.821473183951313, 'pearsonr': 0.6960593797085364}
ttl-160m                {'loss': 6.5078626351356, 'mse': 0.707735346240288, 'pearsonr': 0.7445772730803368}
ttl-460m                {'loss': 13.014084571838, 'mse': 0.777686260176448, 'pearsonr': 0.7184561727175540}

## ASSIN2-RTE

qwen-pt-base-unigram-8k {'accuracy': 0.936, 'f1': 0.9365079365079365, 'loss': 0.365410703573375}
qwen-pt-base-bpe-8k     {'accuracy': 0.946, 'f1': 0.9467455621301775, 'loss': 0.191030851928982}
albertina-100m          {'accuracy': 0.954, 'f1': 0.9540918163672655, 'loss': 0.269388559813262}
bertimbau-base          {'accuracy': 0.966, 'f1': 0.9659318637274549, 'loss': 0.168342445563524}
bertimbau-large         {'accuracy': 0.960, 'f1': 0.9604743083003953, 'loss': 0.210415587301831}
ttl-160m                {'accuracy': 0.940, 'f1': 0.9404761904761905, 'loss': 0.245499517741613}
ttl-460m                {'accuracy': 0.954, 'f1': 0.9544554455445544, 'loss': 0.233077387906610}

## ASSIN2-STS

qwen-pt-base-unigram-8k {'loss': 15.451028581619262, 'mse': 0.9048237733380948, 'pearsonr': 0.7654846764464969}
qwen-pt-base-bpe-8k     {'loss':  7.950329521179199, 'mse': 0.8705843447599575, 'pearsonr': 0.7956520071877217}
albertina-100m          {'loss':  0.137625703811645, 'mse': 0.1376257031247484, 'pearsonr': 0.9580658374710770}
bertimbau-base          {'loss':  7.849811948776245, 'mse': 0.8267263170340283, 'pearsonr': 0.8191358804106317}
bertimbau-large         {'loss':  7.800637733459473, 'mse': 0.8255508279107396, 'pearsonr': 0.8151333621930221}
ttl-160m                {'loss':  7.712226789474487, 'mse': 0.8509338292426646, 'pearsonr': 0.7090601661567368}
ttl-460m                {'loss':  7.706326526641845, 'mse': 0.8378874536424850, 'pearsonr': 0.7463324651720226}

## HATEBR-OFFENSIVE-LANGUAGE

qwen-pt-base-unigram-8k {'f1': 0.9196387570794429, 'loss': 0.204905217679749}
qwen-pt-base-bpe-8k     {'f1': 0.9241056302939711, 'loss': 0.386062242936350}
albertina-100m          {'f1': 0.9061871040772712, 'loss': 0.287028435997698}
bertimbau-base          {'f1': 0.9285486443381180, 'loss': 0.419222071491198}
bertimbau-large         {'f1': 0.9383869139162817, 'loss': 0.309882917755645}
ttl-160m                {'f1': 0.9098213566815663, 'loss': 0.434511037814419}
ttl-460m                {'f1': 0.9330352338305998, 'loss': 0.213013658994944}

## HATEBR-HATE-SPEECH

qwen-pt-base-unigram-8k {'f1': 0.8642913902806035, 'loss': 0.3550173878894254}
qwen-pt-base-bpe-8k     {'f1': 0.8730885755862496, 'loss': 0.3724935657591453}
albertina-100m          {'f1': 0.8351486605828672, 'loss': 0.3849389718912009}
bertimbau-base          {'f1': 0.8504273504273504, 'loss': 0.4704934780851805}
bertimbau-large         
ttl-160m                {'f1': 0.8385492971834383, 'loss': 0.2715505949188290}
ttl-460m                {'f1': 0.8503909427151669, 'loss': 0.2942065685501348}

## PLUE-RTE

qwen-pt-base-unigram-8k {'accuracy': 0.6137184115523465, 'f1': 0.610796685619739, 'loss': 1.270848372917528}
qwen-pt-base-bpe-8k     {'accuracy': 0.6353790613718412, 'f1': 0.634158461156225, 'loss': 2.408486052971024}
albertina-100m          
bertimbau-base          
bertimbau-large         
ttl-160m                {'accuracy': 0.6353790613718412, 'f1': 0.634445271974181, 'loss': 1.1995638508874156}
ttl-460m                


# TEST

## ASSIN-RTE

qwen-pt-base-unigram-8k {'accuracy': 0.864, 'f1': 0.682683201841826, 'loss': 0.4936034766156226}
qwen-pt-base-bpe-8k     {'accuracy': 0.871, 'f1': 0.730912562783493, 'loss': 0.4964878214327618}
albertina-100m          {'accuracy': 0.897, 'f1': 0.808164830145459, 'loss': 0.3497483072672039}
bertimbau-base          {'accuracy': 0.915, 'f1': 0.817268673764008, 'loss': 0.3314594001397490}
bertimbau-large         {'accuracy': 0.909, 'f1': 0.825488487284687, 'loss': 0.4487805549349868}
ttl-160m                {'accuracy': 0.849, 'f1': 0.699824254483373, 'loss': 0.4975565618481487}
ttl-460m                {'accuracy': 0.881, 'f1': 0.727950558579012, 'loss': 0.6702478250308923}

## ASSIN-STS

qwen-pt-base-unigram-8k {'loss': 15.353971364974976, 'mse': 0.8786739621513636, 'pearsonr': 0.73772019140958}
qwen-pt-base-bpe-8k     {'loss': 15.362796483993531, 'mse': 0.8772984295783519, 'pearsonr': 0.74543065891152}
albertina-100m          {'loss':  0.392610147565603, 'mse': 0.3926101470395435, 'pearsonr': 0.82291001522343}
bertimbau-base          {'loss': 15.943377103805542, 'mse': 0.8448226692957302, 'pearsonr': 0.79050414844020}
bertimbau-large         {'loss': 15.752563692092895, 'mse': 0.9805655466281838, 'pearsonr': 0.66167556250696}
ttl-160m                {'loss': 15.351681617736816, 'mse': 0.8477843261034256, 'pearsonr': 0.71109261597264}
ttl-460m                {'loss': 15.347406528472900, 'mse': 0.9195762674169218, 'pearsonr': 0.68104611533434}

## ASSIN2-RTE

qwen-pt-base-unigram-8k {'accuracy': 0.8390522875816994, 'f1': 0.853749072011878, 'loss': 1.1189387508168105}
qwen-pt-base-bpe-8k     {'accuracy': 0.8484477124183006, 'f1': 0.858988977575066, 'loss': 0.6156039649797177}
albertina-100m          {'accuracy': 0.8705065359477124, 'f1': 0.879238095238095, 'loss': 0.8381799205755693}
bertimbau-base          {'accuracy': 0.8937908496732027, 'f1': 0.899146625290923, 'loss': 0.5888805840201706}
bertimbau-large         {'accuracy': 0.8864379084967320, 'f1': 0.894135567402894, 'loss': 0.6726859007559540}
ttl-160m                {'accuracy': 0.8664215686274510, 'f1': 0.872315501757126, 'loss': 0.6125009593111006}
ttl-460m                {'accuracy': 0.8745915032679739, 'f1': 0.879937426671881, 'loss': 0.8166417050983523}

## ASSIN2-STS

qwen-pt-base-unigram-8k {'loss': 18.45679362614949, 'mse': 1.1000377564742734, 'pearsonr': 0.7187441844852063}
qwen-pt-base-bpe-8k     {'loss': 20.86190151389128, 'mse': 1.1857147752391477, 'pearsonr': 0.7635071402227767}
albertina-100m          {'loss': 0.742108191734825, 'mse': 0.7421081916296381, 'pearsonr': 0.7810604970026523}
bertimbau-base          {'loss': 20.01898302128112, 'mse': 1.0911670898993777, 'pearsonr': 0.7903866381359168}
bertimbau-large         {'loss': 19.59321536269842, 'mse': 1.0548054050188990, 'pearsonr': 0.7995862130585414}
ttl-160m                {'loss': 18.72409152672961, 'mse': 1.0767523598427566, 'pearsonr': 0.6119540400841965}
ttl-460m                {'loss': 18.49882795919779, 'mse': 1.0311919258057310, 'pearsonr': 0.6956606368857454}

## HATEBR-OFFENSIVE-LANGUAGE

qwen-pt-base-unigram-8k {'f1': 0.8999997959179509, 'loss': 0.2416319667867252}  * max batch size = 8
qwen-pt-base-bpe-8k     {'f1': 0.9157058562999156, 'loss': 0.3763829595669189}
albertina-100m          {'f1': 0.8905909772800059, 'loss': 0.3108106080974851}  * max batch size = 8
bertimbau-base          {'f1': 0.9142463376040018, 'loss': 0.3902473818134084}
bertimbau-large         {'f1': 0.9249913893176513, 'loss': 0.3360386834582979}
ttl-160m                {'f1': 0.9071155604912464, 'loss': 0.4370277327819661}
ttl-460m                {'f1': 0.9192853436571903, 'loss': 0.2730217934835569}  * max batch size = 8

## HATEBR-HATE-SPEECH

qwen-pt-base-unigram-8k {'f1': 0.8585471422611598, 'loss': 0.327181950655898}   * max batch size = 8
qwen-pt-base-bpe-8k     {'f1': 0.8871149814546041, 'loss': 0.326590922042794}
albertina-100m          {'f1': 0.8428237830070056, 'loss': 0.380224674434534}
bertimbau-base          {'f1': 0.8627912186379928, 'loss': 0.442413617139016}
bertimbau-large         -> script 2
ttl-160m                {'f1': 0.8142602630681173, 'loss': 0.284677883471761}
ttl-460m                {'f1': 0.8683810873747089, 'loss': 0.328360460471761}   * max batch size = 8

## PLUE-RTE

qwen-pt-base-unigram-8k ok
qwen-pt-base-bpe-8k     ok
albertina-100m          -> script 1
bertimbau-base          -> script 3
bertimbau-large         script 5
ttl-160m                ok
ttl-460m                -> script 4