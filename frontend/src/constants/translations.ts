import type { Language, LanguageOption, ExampleQuestion } from '../types';

export const LANGUAGES: LanguageOption[] = [
  { code: 'en', label: 'English', nativeLabel: 'English' },
  { code: 'hi', label: 'Hindi', nativeLabel: 'हिंदी' },
  { code: 'kn', label: 'Kannada', nativeLabel: 'ಕನ್ನಡ' },
];

export interface TranslationStrings {
  appName: string;
  subTitle: string;
  shortDesc: string;
  stepBadge: string;
  backendOnline: string;
  backendOffline: string;
  backendChecking: string;
  inputPlaceholder: string;
  sendButton: string;
  micTooltip: string;
  micModalTitle: string;
  micModalDesc: string;
  examplesTitle: string;
  disclaimerText: string;
  emptyStateTitle: string;
  emptyStateSubtitle: string;
  loadingMessage: string;
  errorSending: string;
  clearChat: string;
}

export const TRANSLATIONS: Record<Language, TranslationStrings> = {
  en: {
    appName: 'NyayaSetu AI',
    subTitle: 'Multilingual Agentic AI-Based Family Law Information and Decision-Support System',
    shortDesc: 'Authoritative guidance for Indian family law — covering maintenance, divorce, child custody, and domestic violence.',
    stepBadge: 'Step 1: Foundation Working',
    backendOnline: 'Backend Online',
    backendOffline: 'Backend Offline',
    backendChecking: 'Connecting...',
    inputPlaceholder: 'Ask a question about maintenance, divorce, child custody, or domestic violence...',
    sendButton: 'Send Query',
    micTooltip: 'Voice input (ASR) placeholder — Scheduled for Step 3',
    micModalTitle: 'Voice Input (Speech-to-Text)',
    micModalDesc: 'Multilingual speech recognition for English, Hindi, and Kannada will be integrated in Step 3. For Step 1, please type your legal query.',
    examplesTitle: 'Example Legal Queries',
    disclaimerText: 'Legal Disclaimer: NyayaSetu AI provides general legal information and decision support based on Indian statutory law. It is not legal advice and is not a substitute for a licensed advocate or qualified legal practitioner.',
    emptyStateTitle: 'How can NyayaSetu AI assist you today?',
    emptyStateSubtitle: 'Select an example query below or type your inquiry in English, Hindi, or Kannada.',
    loadingMessage: 'Consulting NyayaSetu legal pipeline...',
    errorSending: 'Unable to reach backend. Please verify FastAPI is running on http://127.0.0.1:8000.',
    clearChat: 'Clear Conversation',
  },
  hi: {
    appName: 'न्यायसेतु एआई (NyayaSetu AI)',
    subTitle: 'बहुभाषी एजेंटिक एआई आधारित पारिवारिक कानून सूचना एवं निर्णय-सहायता प्रणाली',
    shortDesc: 'भारतीय पारिवारिक कानून हेतु आधिकारिक मार्गदर्शन — भरण-पोषण, तलाक, बाल अभिरक्षा एवं घरेलू हिंसा।',
    stepBadge: 'चरण 1: आधारभूत ढांचा सक्रिय',
    backendOnline: 'सर्वर ऑनलाइन',
    backendOffline: 'सर्वर ऑफलाइन',
    backendChecking: 'जुड़ रहा है...',
    inputPlaceholder: 'भरण-पोषण, तलाक, बाल अभिरक्षा या घरेलू हिंसा से संबंधित प्रश्न पूछें...',
    sendButton: 'प्रश्न भेजें',
    micTooltip: 'आवाज इनपुट पूर्वावलोकन — चरण 3 के लिए नियोजित',
    micModalTitle: 'ध्वनि इनपुट (Speech-to-Text)',
    micModalDesc: 'हिंदी, अंग्रेजी और कन्नड़ के लिए वाक पहचान चरण 3 में जोड़ी जाएगी। चरण 1 के लिए कृपया अपना प्रश्न टाइप करें।',
    examplesTitle: 'उदाहरण कानूनी प्रश्न',
    disclaimerText: 'विधिक अस्वीकरण: न्यायसेतु एआई भारतीय संहिताओं के आधार पर सामान्य कानूनी जानकारी और निर्णय-सहायता प्रदान करता है। यह किसी पंजीकृत अधिवक्ता की कानूनी सलाह का विकल्प नहीं है।',
    emptyStateTitle: 'न्यायसेतु एआई आपकी क्या सहायता कर सकता है?',
    emptyStateSubtitle: 'नीचे दिए गए उदाहरण प्रश्नों में से चुनें या अपना प्रश्न हिंदी, अंग्रेजी अथवा कन्नड़ में लिखें।',
    loadingMessage: 'न्यायसेतु कानूनी पाइपलाइन से संपर्क हो रहा है...',
    errorSending: 'बैकएंड से संपर्क नहीं हो सका। कृपया जांचें कि फास्टएपीआई सर्वर चल रहा है।',
    clearChat: 'बातचीत साफ़ करें',
  },
  kn: {
    appName: 'ನ್ಯಾಯಸೇತು ಎಐ (NyayaSetu AI)',
    subTitle: 'ಬಹುಭಾಷಾ ಏಜೆಂಟಿಕ್ ಎಐ ಆಧಾರಿತ ಕೌಟುಂಬಿಕ ಕಾನೂನು ಮಾಹಿತಿ ಮತ್ತು ನಿರ್ಧಾರ-ಬೆಂಬಲ ವ್ಯವಸ್ಥೆ',
    shortDesc: 'ಭಾರತೀಯ ಕೌಟುಂಬಿಕ ಕಾನೂನಿಗೆ ಅಧಿಕೃತ ಮಾರ್ಗದರ್ಶನ — ಜೀವನಾಂಶ, ವಿಚ್ಛೇದನ, ಮಕ್ಕಳ ಪಾಲನೆ ಮತ್ತು ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ.',
    stepBadge: 'ಹಂತ 1: ಮೂಲಭೂತ ಸಂಪರ್ಕ ಸಿದ್ಧ',
    backendOnline: 'ಸರ್ವರ್ ಆನ್‌ಲೈನ್',
    backendOffline: 'ಸರ್ವರ್ ಆಫ್‌ಲೈನ್',
    backendChecking: 'ಸಂಪರ್ಕಿಸಲಾಗುತ್ತಿದೆ...',
    inputPlaceholder: 'ಜೀವನಾಂಶ, ವಿಚ್ಛೇದನ, ಮಕ್ಕಳ ಪಾಲನೆ ಅಥವಾ ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ ಕುರಿತು ಪ್ರಶ್ನಿಸಿ...',
    sendButton: 'ಪ್ರಶ್ನೆ ಕಳುಹಿಸಿ',
    micTooltip: 'ಧ್ವನಿ ಇನ್‌ಪುಟ್ ಪ್ಲೇಸ್‌ಹೋಲ್ಡರ್ — ಹಂತ 3 ರಲ್ಲಿ ಬರಲಿದೆ',
    micModalTitle: 'ಧ್ವನಿ ಇನ್‌ಪುಟ್ (Speech-to-Text)',
    micModalDesc: 'ಕನ್ನಡ, ಹಿಂದಿ ಮತ್ತು ಇಂಗ್ಲಿಷ್ ಭಾಷೆಗಳಿಗೆ ಧ್ವನಿ ಗುರುತಿಸುವಿಕೆಯನ್ನು ಹಂತ 3 ರಲ್ಲಿ ಸಂಯೋಜಿಸಲಾಗುತ್ತದೆ. ಹಂತ 1 ಕ್ಕೆ ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ.',
    examplesTitle: 'ಮಾದರಿ ಕಾನೂನು ಪ್ರಶ್ನೆಗಳು',
    disclaimerText: 'ಕಾನೂನು ಹಕ್ಕುತ್ಯಾಗ: ನ್ಯಾಯಸೇತು ಎಐ ಭಾರತೀಯ ಕಾನೂನುಗಳ ಆಧಾರದ ಮೇಲೆ ಸಾಮಾನ್ಯ ಕಾನೂನು ಮಾಹಿತಿ ನೀಡುತ್ತದೆ. ಇದು ಅರ್ಹ ವಕೀಲರ ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ.',
    emptyStateTitle: 'ನ್ಯಾಯಸೇತು ಎಐ ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?',
    emptyStateSubtitle: 'ಕೆಳಗಿನ ಮಾದರಿ ಪ್ರಶ್ನೆಗಳಲ್ಲಿ ಒಂದನ್ನು ಆಯ್ಕೆಮಾಡಿ ಅಥವಾ ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ.',
    loadingMessage: 'ನ್ಯಾಯಸೇತು ಕಾನೂನು ಪೈಪ್‌ಲೈನ್ ಸಂಪರ್ಕಿಸಲಾಗುತ್ತಿದೆ...',
    errorSending: 'ಸರ್ವರ್ ಸಂಪರ್ಕಿಸಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು FastAPI ಚಾಲನೆಯಲ್ಲಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.',
    clearChat: 'ಸಂಭಾಷಣೆ ಅಳಿಸಿ',
  },
};

export const EXAMPLE_QUESTIONS: Record<Language, ExampleQuestion[]> = {
  en: [
    {
      id: 'maint_en',
      category: 'maintenance',
      categoryLabel: 'Maintenance',
      question: 'What are the legal guidelines for interim maintenance under Section 125 of CrPC / Section 144 of BNSS?',
    },
    {
      id: 'div_en',
      category: 'divorce',
      categoryLabel: 'Divorce',
      question: 'What is the required cooling-off period and procedure for mutual consent divorce under the Hindu Marriage Act, 1955?',
    },
    {
      id: 'cust_en',
      category: 'custody',
      categoryLabel: 'Child Custody',
      question: 'How do Indian family courts evaluate the "welfare of the minor child" principle in custody disputes?',
    },
    {
      id: 'dv_en',
      category: 'domestic_violence',
      categoryLabel: 'Domestic Violence',
      question: 'What emergency protection and residence orders can an aggrieved woman seek under the DV Act, 2005?',
    },
  ],
  hi: [
    {
      id: 'maint_hi',
      category: 'maintenance',
      categoryLabel: 'भरण-पोषण',
      question: 'सीआरपीसी धारा 125 (बीएनएसएस 144) के अंतर्गत अंतरिम भरण-पोषण तय करने के क्या कानूनी मापदंड हैं?',
    },
    {
      id: 'div_hi',
      category: 'divorce',
      categoryLabel: 'विवाह विच्छेद (तलाक)',
      question: 'हिंदू विवाह अधिनियम 1955 के तहत आपसी सहमति से तलाक की प्रक्रिया और कूलिंग-ऑफ अवधि क्या है?',
    },
    {
      id: 'cust_hi',
      category: 'custody',
      categoryLabel: 'बाल अभिरक्षा',
      question: 'कस्टडी विवादों में पारिवारिक अदालतें "बच्चे के सर्वोत्तम कल्याण" के सिद्धांत का मूल्यांकन कैसे करती हैं?',
    },
    {
      id: 'dv_hi',
      category: 'domestic_violence',
      categoryLabel: 'घरेलू हिंसा',
      question: 'घरेलू हिंसा से महिला संरक्षण अधिनियम 2005 के तहत कौन से आपातकालीन संरक्षण व निवास आदेश प्राप्त किए जा सकते हैं?',
    },
  ],
  kn: [
    {
      id: 'maint_kn',
      category: 'maintenance',
      categoryLabel: 'ಜೀವನಾಂಶ',
      question: 'ಸಿಆರ್‌ಪಿಸಿ ಸೆಕ್ಷನ್ 125 ರ ಅಡಿಯಲ್ಲಿ ಮಧ್ಯಂತರ ಜೀವನಾಂಶ ನಿರ್ಧರಿಸಲು ಕಾನೂನು ಮಾರ್ಗಸೂಚಿಗಳು ಯಾವುವು?',
    },
    {
      id: 'div_kn',
      category: 'divorce',
      categoryLabel: 'ವಿಚ್ಛೇದನ',
      question: 'ಹಿಂದೂ ವಿವಾಹ ಕಾಯ್ದೆ 1955 ರ ಪ್ರಕಾರ ಪರಸ್ಪರ ಒಪ್ಪಿಗೆಯ ವಿಚ್ಛೇದನದ ಪ್ರಕ್ರಿಯೆ ಮತ್ತು ಕಾಯುವ ಅವಧಿ ಏನು?',
    },
    {
      id: 'cust_kn',
      category: 'custody',
      categoryLabel: 'ಮಕ್ಕಳ ಪಾಲನೆ',
      question: 'ಮಕ್ಕಳ ಪಾಲನೆ ವಿವಾದಗಳಲ್ಲಿ ಕೌಟುಂಬಿಕ ನ್ಯಾಯಾಲಯಗಳು "ಮಗುವಿನ ಕ್ಷೇಮಾಭಿವೃದ್ಧಿ" ತತ್ವವನ್ನು ಹೇಗೆ ಮೌಲ್ಯಮಾಪನ ಮಾಡುತ್ತವೆ?',
    },
    {
      id: 'dv_kn',
      category: 'domestic_violence',
      categoryLabel: 'ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ',
      question: 'ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ ತಡೆ ಕಾಯ್ದೆ 2005 ರ ಅಡಿಯಲ್ಲಿ ಮಹಿಳೆಯು ಪಡೆಯಬಹುದಾದ ತುರ್ತು ರಕ್ಷಣೆ ಮತ್ತು ವಾಸದ ಆದೇಶಗಳು ಯಾವುವು?',
    },
  ],
};
