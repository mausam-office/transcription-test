CACHE_DIR = "./cache"
LANGUAGE = "Nepali"
TASK = "transcribe"
N_MIDDLE_WORDS = 2
PRETRAINED_MODEL_NAME = "exps-whisper-small-transcriber/exp-3-new-words/checkpoint-6009"
MAX_AUDIO_DURATION = 25.0

KODES = [
    "अखण्ड",
    "अगुवा",
    "अग्रज",
    "अजय",
    "अजर",
    "अटल",
    "अतिथी",
    "अतुल",
    "अथाह",
    "अनन्त",
    "अनुज",
    "अपार",
    "अपुर्ब",
    "अभय",
    "अमर",
    "अमल",
    "अरुण",
    "अर्पण",
    "असल",
    "अक्षर",
    "आँगन",
    "आँचल",
    "आकार",
    "आकाश",
    "आदर",
    "आदर्श",
    "आधार",
    "आमूल",
    "आयाम",
    "आर्जन",
    "आलोक",
    "आवास",
    "आशा",
    "आसन",
    "आस्था",
    "इनाम",
    "उत्तम",
    "उदय",
    "उदार",
    "उन्नती",
    "उमंग",
    "उषा",
    "ऊर्जा",
    "एकता",
    "कण्ठ",
    "कदम",
    "कदर",
    "कपिल",
    "कपुर",
    "कमल",
    "करुणा",
    "कलम",
    "कला",
    "कवि",
    "कविता",
    "कसम",
    "काब्य",
    "कामना",
    "किनारा",
    "किरण",
    "किल्ला",
    "कीर्ति",
    "कुञ्ज",
    "कुशल",
    "कुसुम",
    "कृपा",
    "केशर",
    "कोइली",
    "कोपिला",
    "खजाना",
    "खनिज",
    "खानी",
    "गगन",
    "गमला",
    "गहना",
    "गाजल",
    "गाथा",
    "गुराँस",
    "गृह",
    "गोचर",
    "गोरस",
    "गोरेटो",
    "गौँथली",
    "गौरव",
    "घुम्ती",
    "चक्र",
    "चन्द्र",
    "चमक",
    "चमेली",
    "चरण",
    "चाँदनी",
    "चाँदी",
    "चाहना",
    "चित्र",
    "चिराग",
    "चिरायु",
    "चेतन",
    "चेहरा",
    "चौतारी",
    "छवि",
    "छहारी",
    "छाता",
    "छायाँ",
    "छिमेकी",
    "जगत्",
    "जनता",
    "जमिन",
    "जमुना",
    "जरायो",
    "जल",
    "जहाज",
    "जागृति",
    "जाती",
    "जीवन",
    "जुनेली",
    "जुरेली",
    "जुही",
    "जोर",
    "ज्योति",
    "झरना",
    "झलक",
    "डबली",
    "डाँफे",
    "डाली",
    "ढोका",
    "तक्मा",
    "तपन",
    "तबला",
    "तरल",
    "तराजु",
    "तरेली",
    "तलाउ",
    "तलैया",
    "तारा",
    "ताली",
    "तिर्सना",
    "तेज",
    "तोरण",
    "त्रिकोण",
    "थलो",
    "थाली",
    "थुँगो",
    "थैली",
    "दर्पण",
    "दर्शन",
    "दलान",
    "दियालो",
    "दिल",
    "दिवस",
    "दिवा",
    "दीक्षा",
    "दीप",
    "दोभान",
    "द्रव्य",
    "धन",
    "धनेश",
    "धैर्य",
    "नक्षत्र",
    "नगर",
    "नजिर",
    "नभ",
    "नमन",
    "नमुना",
    "नयन",
    "नरम",
    "नहर",
    "नाका",
    "निकट",
    "निकास",
    "निगम",
    "निवास",
    "नियम",
    "निर्दोष",
    "निर्मल",
    "निर्माण",
    "निपुण",
    "निसाना",
    "नेत्र",
    "न्याय",
    "पठन",
    "पन्ना",
    "परी",
    "परेली",
    "परेवा",
    "पवन",
    "पवित्र",
    "पत्रिका",
    "पाइला",
    "पिपल",
    "पुष्प",
    "पुस्तक",
    "पृथ्वी",
    "पोखरी",
    "पोषण",
    "पौवा",
    "प्रकाश",
    "प्रकृति",
    "प्रखर",
    "प्रगति",
    "प्रणय",
    "प्रदीप",
    "प्रबल",
    "प्रभाव",
    "प्रयास",
    "प्रीति",
    "प्रेम",
    "प्रज्ञा",
    "फटिक",
    "फूल",
    "बगल",
    "बगान",
    "बगैँचा",
    "बचत",
    "बचन",
    "बटुवा",
    "बर",
    "बहाल",
    "बाला",
    "बिगुल",
    "बिरुवा",
    "भन्डार",
    "भरोसा",
    "भवन",
    "भुवन",
    "भूगोल",
    "भूमिका",
    "मंगल",
    "मकल",
    "मखन",
    "मगन",
    "मञ्जरी",
    "मणि",
    "मदत",
    "मदन",
    "मधु",
    "मन",
    "मनीषा",
    "ममता",
    "मयूर",
    "मलम",
    "मसला",
    "मह",
    "महल",
    "महिमा",
    "माणिक",
    "मानक",
    "मानव",
    "मायालु",
    "माला",
    "माली",
    "मिठास",
    "मिलन",
    "मिसिल",
    "मीत",
    "मुना",
    "मुनाल",
    "मुरली",
    "मुलुक",
    "मुहान",
    "मुहार",
    "मैदान",
    "मैना",
    "मोती",
    "मोहर",
    "मौरी",
    "मौलिक",
    "मौसम",
    "यन्त्र",
    "यात्री",
    "युगल",
    "युवा",
    "योग",
    "योजना",
    "रचना",
    "रजत",
    "रत्न",
    "रमण",
    "रमा",
    "रवि",
    "रश्मि",
    "रसिक",
    "रिवाज",
    "रुपक",
    "रेखा",
    "रेसम",
    "रोचक",
    "रोहिणी",
    "रौनक",
    "लगन",
    "लता",
    "लवण",
    "लहना",
    "लहर",
    "लक्ष",
    "लाँकुरी",
    "लाभ",
    "लाली",
    "लेखक",
    "लोचन",
    "वजन",
    "विकास",
    "विजय",
    "विज्ञान",
    "विद्या",
    "विधान",
    "विनम्र",
    "विनय",
    "विपना",
    "विपुल",
    "विविध",
    "विवेक",
    "विशाल",
    "विहानी",
    "वृक्ष",
    "शयन",
    "शान",
    "शान्ति",
    "शालीन",
    "शिक्षा",
    "शिर",
    "शिशिर",
    "शीतल",
    "शुभ",
    "शैली",
    "संगम",
    "संयोग",
    "सकल",
    "सक्षम",
    "सखा",
    "सगुन",
    "सगोल",
    "सघन",
    "सचेत",
    "सजग",
    "सटीक",
    "सतह",
    "सत्य",
    "सदन",
    "सन्दुक",
    "सन्ध्या",
    "सपना",
    "सफल",
    "सफा",
    "सफेद",
    "समता",
    "समय",
    "समाज",
    "समीप",
    "समीर",
    "समूह",
    "सरल",
    "सवल",
    "सहारा",
    "साँध",
    "साइत",
    "साकार",
    "साख",
    "सागर",
    "साझा",
    "साथी",
    "सादा",
    "साधना",
    "सारथी",
    "साहस",
    "साक्षी",
    "सिन्धु",
    "सिप",
    "सिमल",
    "सिमाना",
    "सिम्रिक",
    "सिरानी",
    "सिर्जना",
    "सुख",
    "सुगन्ध",
    "सुगम",
    "सुदिन",
    "सुदीप",
    "सुधार",
    "सुन",
    "सुनौलो",
    "सुन्दर",
    "सुमन",
    "सुमार्ग",
    "सुमुख",
    "सुयोग",
    "सुरज",
    "सुरक्षा",
    "सुलभ",
    "सुवर्ण",
    "सुविधा",
    "सुशील",
    "सेवा",
    "सौगात",
    "हरिण",
    "हरित",
    "हवेली",
    "हाकिम",
    "हार्दिक",
    "क्षमता",
    "ज्ञान",
]

DIGITS = [
    "शून्य",
    "एक",
    "दुई",
    "तीन",
    "चार",
    "पाँच",
    "छ",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "एघार",
    "बाह्र",
    "तेह्र",
    "चौध",
    "पन्ध्र",
    "सोह्र",
    "सत्र",
    "अठार",
    "उन्नाइस",
    "बिस",
    "एक्काइस",
    "बाइस",
    "तेइस",
    "चौबिस",
    "पच्चिस",
    "छब्बिस",
    "सत्ताइस",
    "अठ्ठाइस",
    "उनन्तिस",
    "तिस",
    "एकतिस",
    "बत्तिस",
    "तेत्तिस",
    "चौतिस",
    "पैँतिस",
    "छत्तिस",
    "सैँतिस",
    "अठतिस",
    "उनन्चालिस",
    "चालिस",
    "एकचालिस",
    "बयालिस",
    "त्रिचालिस",
    "चवालिस",
    "पैँतालिस",
    "छयालिस",
    "सतचालिस",
    "अठचालिस",
    "उनन्चास",
    "पचास",
    "एकाउन्न",
    "बाउन्न",
    "त्रिपन्न",
    "चवन्न",
    "पचपन्न",
    "छपन्न",
    "सन्ताउन्न",
    "अन्ठाउन्न",
    "उनसट्ठी",
    "साठी",
    "एकसट्ठी",
    "बयसट्ठी",
    "त्रिसट्ठी",
    "चौसट्ठी",
    "पैँसट्ठी",
    "छयसट्ठी",
    "सतसट्ठी",
    "अठसट्ठी",
    "उनन्सत्तरी",
    "सत्तरी",
    "एकहत्तर",
    "बहत्तर",
    "त्रिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छयहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उनासी",
    "असी",
    "एकासी",
    "बयासी",
    "त्रियासी",
    "चौरासी",
    "पचासी",
    "छयासी",
    "सतासी",
    "अठासी",
    "उनान्नब्बे",
    "नब्बे",
    "एकान्नब्बे",
    "बयान्नब्बे",
    "त्रियान्नब्बे",
    "चौरान्नब्बे",
    "पन्चान्नब्बे",
    "छयान्नब्बे",
    "सन्तान्नब्बे",
    "अन्ठान्नब्बे",
    "उनान्सय",
    "सय",
    "हजार",
]
