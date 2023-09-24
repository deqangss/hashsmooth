import os
import warnings
import argparse

from pprint import pprint
import numpy as np
from sklearn.model_selection import train_test_split
import re
import collections
from androguard.misc import APK, AnalyzeAPK
import lxml.etree as etree
import multiprocessing
from xml.dom import minidom
from sys import platform as _platform

from tools import utils
from sec_classifiers.drebin import drebin_utils

logger = utils.logging.getLogger("Get-Drebin-Features")
logger.addHandler(utils.ErrorHandler)

cmd_md = argparse.ArgumentParser(description='arguments for drebin feature extraction')
cmd_md.add_argument('--dataset_dir', type=str, default='',
                    help='Folder path to dataset directory.')
cmd_md.add_argument('--feature_dir', type=str, default='',
                    help='Folder path to saving feature files.')
cmd_md.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite the existing feature files.')

if _platform == "linux" or _platform == "linux2":
    TMP_DIR = '/tmp/'
elif _platform == "win32" or _platform == "win64":
    TMP_DIR = 'C:\\TEMP\\'
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

current_dir = os.path.dirname(os.path.realpath(__file__))
API_SPLITER = ':::'
# information about feature extraction
SuspiciousNames = ["getExternalStorageDirectory",
                   "getSimCountryIso",
                   "execHttpRequest",
                   "sendTextMessage",
                   "getMessageBody",
                   "getPackageInfo",
                   "getSystemService",
                   "setWifiDisabled",
                   "Cipher",
                   "crypto",
                   "Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;)",
                   "Lorg/apache/http/client/methods/HttpPost",
                   "Ljava/io/IOException;->printStackTrace",
                   "Ljava/lang/Runtime;->exec",
                   "Ljava/lang/System;->loadLibrary",
                   "Ljava/lang/System;->load",
                   "Ldalvik/system/DexClassLoader;",
                   "Ldalvik/system/SecureClassLoader;",
                   "Ldalvik/system/PathClassLoader;",
                   "Ldalvik/system/BaseDexClassLoader;",
                   "Ldalvik/system/URLClassLoader;",
                   "android/os/Exec",
                   "Base64",
                   "system/bin/su"
                   ]

COMP = {
    "Permission": "permission",
    "Activity": "activity",
    "Service": "service",
    "Receiver": "receiver",
    "Provider": "provider",
    "Hardware": "hardware",
    "Intentfilter": 'intent-filter',
    "Android_API": "android_api",
    "Java_API": "java_api",
    "User_String": "const-string",
    "User_Class": "user_class",
    "User_Method": "user_method",
    "OpCode": "opcode",
    "Asset": "asset",
    "Notdefined": 'not_defined'
}

OPERATOR = {
    # insert
    0: "insert",
    # remove
    1: "remove"
}

INSTR_ALLOWED = {
    OPERATOR[0]: [COMP['Permission'],
                  COMP['Activity'],
                  COMP['Service'],
                  COMP['Receiver'],
                  COMP['Hardware'],
                  COMP['Intentfilter'],
                  COMP['Android_API'],
                  COMP['User_String']
                  ],
    OPERATOR[1]: [COMP['Activity'],
                  COMP['Service'],
                  COMP['Receiver'],
                  COMP['Provider'],
                  COMP['Android_API'],
                  COMP['User_String']
                  ]
}


def get_drebin(pargs):
    apk_path, feature_save_path, overwrite = pargs
    try:
        assert os.path.exists(apk_path)
        if not overwrite and os.path.exists(feature_save_path):
            return 1, feature_save_path
        data_dict = {}
        # get xml features
        requested_permission_list, \
        activity_list, \
        service_list, \
        content_provider_list, \
        broadcast_receiver_list, \
        hardware_list, \
        intentfilter_list = get_feature_xml(apk_path)

        # get dex features
        pmap = AxplorerMapping()
        used_permission_list, \
        restricted_api_list, \
        suspicious_api_list, \
        url_list = get_feature_dex(apk_path, pmap, requested_permission_list)
        data_dict['requested_permission_list'] = requested_permission_list
        data_dict['activity_list'] = activity_list
        data_dict['service_list'] = service_list
        data_dict['content_provider_list'] = content_provider_list
        data_dict['broadcast_receiver_list'] = broadcast_receiver_list
        data_dict['hardware_list'] = hardware_list
        data_dict['intentfilter_list'] = intentfilter_list
        data_dict['used_permission_list'] = used_permission_list
        data_dict['restricted_api_list'] = restricted_api_list
        data_dict['suspicious_api_list'] = suspicious_api_list
        data_dict['url_list'] = url_list

        drebin_utils.dump_feature(data_dict, feature_save_path)
        return 1, feature_save_path
    except Exception as e:
        e.args += (apk_path,)
        return 0, e


def get_feature_xml(apk_path):
    """
    get requested feature from manifest file
    :param apk_path: absolute path of an apk file
    :return: tuple of lists
    """
    requested_permission_list = []
    activity_list = []
    service_list = []
    content_provider_list = []
    broadcast_receiver_list = []
    hardware_list = []
    intentfilter_list = []
    xml_tmp_dir = os.path.join(TMP_DIR, 'xml_dir')
    if not os.path.exists(xml_tmp_dir):
        os.mkdir(xml_tmp_dir)
    apk_name = os.path.splitext(os.path.basename(apk_path))[0]
    try:
        apk_path = os.path.abspath(apk_path)
        a = APK(apk_path)
        f = open(os.path.join(xml_tmp_dir, apk_name + '.xml'), 'wb')
        xmlstreaming = etree.tostring(a.xml['AndroidManifest.xml'], pretty_print=True, encoding='utf-8')
        f.write(xmlstreaming)
        f.close()
    except Exception as e:
        raise Exception("Fail to load xml file of apk {}:{}".format(apk_path) + str(e))

    # start obtain feature S1, S2, S3, S4
    try:
        with open(os.path.join(xml_tmp_dir, apk_name + '.xml'), 'rb') as f:
            dom_xml = minidom.parse(f)
            dom_elements = dom_xml.documentElement

            dom_permissions = dom_elements.getElementsByTagName('uses-permission')
            for permission in dom_permissions:
                if permission.hasAttribute('android:name'):
                    requested_permission_list.append(permission.getAttribute('android:name'))

            dom_activities = dom_elements.getElementsByTagName('activity')
            for activity in dom_activities:
                if activity.hasAttribute('android:name'):
                    activity_list.append(activity.getAttribute('android:name'))

            dom_services = dom_elements.getElementsByTagName("service")
            for service in dom_services:
                if service.hasAttribute("android:name"):
                    service_list.append(service.getAttribute("android:name"))

            dom_contentproviders = dom_elements.getElementsByTagName("provider")
            for provider in dom_contentproviders:
                if provider.hasAttribute("android:name"):
                    content_provider_list.append(provider.getAttribute("android:name"))

            dom_broadcastreceivers = dom_elements.getElementsByTagName("receiver")
            for receiver in dom_broadcastreceivers:
                if receiver.hasAttribute("android:name"):
                    broadcast_receiver_list.append(receiver.getAttribute("android:name"))

            dom_hardwares = dom_elements.getElementsByTagName("uses-feature")
            for hardware in dom_hardwares:
                if hardware.hasAttribute("android:name"):
                    hardware_list.append(hardware.getAttribute("android:name"))

            dom_intentfilter_actions = dom_elements.getElementsByTagName("action")
            for action in dom_intentfilter_actions:
                if action.hasAttribute("android:name"):
                    intentfilter_list.append(action.getAttribute("android:name"))

            return requested_permission_list, activity_list, service_list, content_provider_list, broadcast_receiver_list, hardware_list, intentfilter_list
    except Exception as e:
        raise Exception("Fail to process xml file of apk {}:{}".format(apk_path, str(e)))


def get_feature_dex(apk_path, pmap, requested_permission_list):
    """
    get requested feature from .dex files
    :param apk_path: an absolute path of an apk
    :param pmap: PScout mapping
    :param requested_permission_list: a list of permissions
    :return: tupe of lists
    """
    used_permission_list = []
    restricted_api_list = []
    suspicious_api_list = []
    url_list = []
    try:
        apk_path = os.path.abspath(apk_path)
        a, dd, dx = AnalyzeAPK(apk_path)
    except Exception as e:
        raise Exception("Fail to load 'dex' files of apk {}:{} ".format(apk_path, str(e)))

    if not isinstance(dd, list):
        dd = [dd]  # may accommodate multiple dex files
    try:
        for i, d in enumerate(dd):
            for mtd in d.get_methods():
                dex_content = dx.get_method(mtd)
                for basic_block in dex_content.get_basic_blocks().get():
                    dalvik_code_list = []
                    for instruction in basic_block.get_instructions():
                        # dalvik code + performed body (api + arguments + return type)
                        code_line = instruction.get_name() + ' ' + instruction.get_output()
                        dalvik_code_list.append(code_line)
                    apis, suspicious_apis = get_specific_api(dalvik_code_list)
                    used_permissions, restricted_apis = get_permission_and_apis(apis,
                                                                                pmap,
                                                                                requested_permission_list,
                                                                                suspicious_apis)
                    used_permission_list.extend(used_permissions)
                    restricted_api_list.extend(restricted_apis)
                    suspicious_api_list.extend(suspicious_apis)

                    for code_line in dalvik_code_list:
                        url_search = re.search(
                            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                            code_line,
                            re.IGNORECASE)
                        if url_search:
                            url = url_search.group()
                            url_domain = re.sub(r'(.*://)?([^/?]+).*', '\g<1>\g<2>', url)
                            url_list.append(url_domain)
        # remove duplication
        used_permission_list = list(set(used_permission_list))
        restricted_api_list = list(set(restricted_api_list))
        suspicious_api_list = list(set(suspicious_api_list))
        url_list = list(set(url_list))
        return used_permission_list, restricted_api_list, suspicious_api_list, url_list
    except Exception as e:
        raise Exception("Fail to process 'dex' files of apk {}:{}".format(apk_path, str(e)))


def get_specific_api(dalvik_code_list):
    """
    get invoked apis
    :param dalvik_code_list: a list of dalvik codes (line by line)
    :return: list of apis and list of suspicious apis
    """
    api_list = []
    suspicious_api_list = []

    for code_line in dalvik_code_list:
        if 'invoke-' in code_line:
            sub_parts = code_line.split(',')
            for part in sub_parts:
                if ';->' in part:
                    part = part.strip()
                    if part.startswith('Landroid'):
                        entire_api = part
                        api_parts = part.split(';->')
                        api_class = api_parts[0].strip()
                        api_name = api_parts[1].split('(')[0].strip()
                        api_dict = {'entire_api': entire_api, 'api_class': api_class, 'api_name': api_name}
                        api_list.append(api_dict)
                        if api_name in SuspiciousNames:
                            suspicious_api_list.append(api_class + '.' + api_name)
                for e in suspicious_api_list:
                    if e in part:
                        suspicious_api_list.append(e)

        for e in suspicious_api_list:
            if e in code_line:
                suspicious_api_list.append(e)

    # remove duplication
    suspicious_api_list = list(set(suspicious_api_list))

    return api_list, suspicious_api_list


def get_permission_and_apis(apis, pmap, requested_permission_list, suspicious_apis):
    """
    used permission and apis
    :param apis: a list of apis
    :param pmap: pscout mapping
    :param requested_permission_list: a list of permission
    :param suspicious_apis: a list of apis
    :return: used permission, restricted apis
    """
    used_permission_list = []
    restricted_api_list = []
    for api in apis:
        api_class = api['api_class'].replace('/', '.').replace("Landroid", "android").strip()
        permission = pmap.GetPermFromApi(api_class, api['api_name'])
        if permission is not None:
            if (permission in requested_permission_list) and (len(requested_permission_list) > 0):
                used_permission_list.append(permission)
                api_info = api_class + '.' + api['api_name'] + API_SPLITER + api['entire_api']
                if api_info not in suspicious_apis:
                    restricted_api_list.append(api_info)
            else:
                api_info = api_class + '.' + api['api_name'] + API_SPLITER + api['entire_api']
                if api_info not in suspicious_apis:
                    restricted_api_list.append(api_info)
    # remove duplication
    used_permission_list = list(set(used_permission_list))
    restricted_api_list = list(set(restricted_api_list))

    return used_permission_list, restricted_api_list


def wrapper_load_features(path):
    try:
        return drebin_utils.load_feature(path)
    except Exception as e:
        return e


class AxplorerMapping(object):
    def __init__(self):
        with open(os.path.join(current_dir, 'res/axplorerPermApi22Mapping.json'), 'rb') as FH:
            # Use SmallCase json file to prevent run time case conversion in GetPermFromApi
            import json
            self.PermApiDictFromJsonTemp = json.load(FH)
            self.PermApiDictFromJson = {}
            for Perms in self.PermApiDictFromJsonTemp:
                for Api in range(len(self.PermApiDictFromJsonTemp[Perms])):
                    ApiName = self.PermApiDictFromJsonTemp[Perms][Api][0].lower() + \
                              self.PermApiDictFromJsonTemp[Perms][Api][1].lower()
                    '''Exchange key and values inside the dictionary.'''
                    self.PermApiDictFromJson[ApiName] = Perms
        del self.PermApiDictFromJsonTemp

    def GetAllPerms(self):
        return list(self.PermApiDictFromJson.keys())

    def GetAllApis(self):
        return list(self.PermApiDictFromJson.values())

    def GetApisFromPerm(self, Perm):
        PermAsKey = Perm
        if PermAsKey not in self.PermApiDictFromJson:
            logger.error("Permission %s not found in the PScout Dict",
                         PermAsKey)
            return -1
        else:
            return self.PermApiDictFromJson[PermAsKey]

    def GetPermFromApi(self, ApiClass, ApiMethodName):
        ApiClass = ApiClass.lower()
        ApiMethodName = ApiMethodName.lower()

        ApiName = ApiClass + ApiMethodName
        if (ApiClass + ApiMethodName) in self.PermApiDictFromJson:
            return self.PermApiDictFromJson[ApiName]
        else:
            return None

    def PrintDict(self):
        pprint(self.PermApiDictFromJson)

    def PrintAllPerms(self):
        for PermAsKey in self.PermApiDictFromJson:
            print(PermAsKey)

    def PrintAllApis(self):
        for Api in self.PermApiDictFromJson.values():
            print(Api)

    def PrintApisForPerm(self, Perm):
        PermAsKey = Perm

        if PermAsKey not in self.PermApiDictFromJson:
            warnings.warn("Permission {} not found in the PScout Dict".format(
                PermAsKey))
            return -1

        for Api in self.PermApiDictFromJson[Perm]:
            pprint(Api)
        return 0

    ##################################################
    #                 Sorting the dict               #
    ##################################################
    def SortDictByKeys(self):
        self.PermApiDictFromJson = \
            collections.OrderedDict(sorted(self.PermApiDictFromJson.items()))


def get_drebin_features(file_dir=None, feature_save_dir=None, overwrite=True) -> list:
    if file_dir is None or feature_save_dir is None:
        raise ValueError("No files.\n")
    if not os.path.exists(feature_save_dir):
        utils.mkdir(feature_save_dir)

    apk_paths = list(utils.retrive_files_set(file_dir, '', '.apk|'))
    feature_files = []
    pargs = [(apk_path, os.path.join(feature_save_dir, os.path.splitext(os.path.basename(apk_path))[0]), overwrite) for
             apk_path in apk_paths]
    n_proc = 1 if multiprocessing.cpu_count() - 2 <= 1 else multiprocessing.cpu_count() - 2
    with multiprocessing.Pool(n_proc) as pool:
        for idx, res in enumerate(pool.imap(get_drebin, pargs)):
            if res[0]:
                feature_files.append(res[1])
            else:
                print(res[1])
                continue
    # feature_files = []
    # for i, apk_path in enumerate(apk_paths):
    #     feature_save_path = os.path.join(feature_save_dir, os.path.splitext(os.path.basename(apk_path))[0])
    #     res = get_drebin(apk_path, feature_save_path)
    #     if res[0]:
    #         feature_files.append(res[1])
    #     else:
    #         print(res)
    return feature_files


def load_features(feature_path_list):
    """
    load features
    :param feature_path_list: feature paths produced by the method of feature_extraction
    :return: a list of features
    """
    feature_list = []
    n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(n_proc)
    for res in pool.imap(wrapper_load_features, feature_path_list):
        if not isinstance(res, Exception):
            feature_list.append(res)
        else:
            print(str(res))
    return feature_list


def feature_selection(train_features, train_y, vocab, vocab_info, dim):
    """
    feature selection
    :param train_features: 2D feature
    :type train_features: numpy object
    :param train_y: ground truth labels
    :param vocab: a list of words (i.e., features)
    :param dim: the number of remained words
    :param vocab_info: word information
    :return: chose vocab
    """
    is_malware = (train_y == 1)
    mal_features = np.array(train_features, dtype=object)[is_malware]
    ben_features = np.array(train_features, dtype=object)[~is_malware]

    if (len(mal_features) <= 0) or (len(ben_features) <= 0):
        return vocab

    mal_representations = get_feature_representation(mal_features, vocab)
    mal_frequency = np.sum(mal_representations, axis=0) / float(len(mal_features))
    ben_representations = get_feature_representation(ben_features, vocab)
    ben_frequency = np.sum(ben_representations, axis=0) / float(len(ben_features))

    # eliminate the words showing zero occurrence in apk files
    is_null_feature = np.all(mal_representations == 0, axis=0) & np.all(ben_representations, axis=0)
    mal_representations, ben_representations = None, None
    vocab_filtered = list(np.array(vocab)[~is_null_feature])
    # get vocab information
    vocab_info_filtered = collections.defaultdict(set, {k: v for k, v in vocab_info.items() if k in vocab_filtered})

    if len(vocab_filtered) <= dim:
        return vocab_filtered, vocab_info_filtered
    else:
        feature_frq_diff = np.abs(mal_frequency[~is_null_feature] - ben_frequency[~is_null_feature])
        position_flag = np.argsort(feature_frq_diff)[::-1][:dim]

        vocab_selected = []
        vocab_info_selected = collections.defaultdict(set)
        for p in position_flag:
            w = vocab_filtered[p]
            vocab_selected.append(w)
            vocab_info_selected[w] = vocab_info_filtered[w]
        return vocab_selected, vocab_info_selected


def load_vocabulary():
    vocab_path = os.path.join(current_dir, 'res/drebin.vocab')
    if not os.path.exists(vocab_path):
        raise ValueError("A vocabulary is needed.")
    vocab = utils.read_pickle(vocab_path)
    return vocab


def get_vocabulary(feature_list, n=300000):
    """
    obtain the vocabulary based on the feature
    :param feature_list: 2D list of naive feature
    :param n: the number of top frequency items
    :return: feature vocabulary
    """
    c = collections.Counter()
    d = collections.defaultdict(set)
    clean_feature_set = []
    for features in feature_list:
        clean_feature = []
        for feature in features:
            elements = feature.strip().split(API_SPLITER)
            if len(elements) == 0:
                raise ValueError("Null feature.")
            elif len(elements) == 1:
                c[elements[0]] = c[elements[0]] + 1
                d[elements[0]].add(elements[0].split(drebin_utils.K_V_SPLITER, 1)[1])
                clean_feature.append(elements[0])
            elif len(elements) == 2:
                c[elements[0]] = c[elements[0]] + 1
                d[elements[0]].add(elements[1])
                clean_feature.append(elements[0])
            else:
                raise ValueError("Unexpected feature '{}'".format(feat))
        clean_feature_set.append(clean_feature)
    vocab, count = zip(*c.most_common(n))
    return list(vocab), d, clean_feature_set


def get_feature_simf(feature_list):
    """
    obtain the vocabulary based on the feature
    :param feature_list: 2D list of naive feature
    :param n: the number of top frequency items
    :return: feature vocabulary
    """
    clean_feature_set = []
    for features in feature_list:
        clean_feature = []
        for feature in features:
            elements = feature.strip().split(API_SPLITER)
            if len(elements) == 0:
                raise ValueError("Null feature.")
            elif len(elements) == 1:
                clean_feature.append(elements[0])
            elif len(elements) == 2:
                clean_feature.append(elements[0])
            else:
                raise ValueError("Unexpected feature '{}'".format(feat))
        clean_feature_set.append(clean_feature)
    return clean_feature_set


def get_feature_representation(feature_list, vocab):
    """
    mapping feature to numerical representation
    :param feature_list: 2D feature list with shape [number of files, number of feature]
    :param vocab: a list of words
    :return: 2D representation
    :rtype numpy.ndarray
    """
    N = len(feature_list)
    M = len(vocab)

    assert N > 0 and M > 0

    representations = np.zeros((N, M), dtype=np.float32)
    dictionary = dict(zip(vocab, range(len(vocab))))
    for i, features in enumerate(feature_list):
        if len(features) > 0:
            filled_positions = [idx for idx in list(map(dictionary.get, features)) if idx is not None]
            if len(filled_positions) != 0:
                representations[i, filled_positions] = 1.
            else:
                warnings.warn("Produce zero feature vector.")

    return representations


def feature_preprocess(feature_path_list, gt_labels, feature_save_dir, overwrite=True):
    """
    pre-processing the naive data to accommodate the input format of ML algorithms
    :param feature_path_list: feature paths produced by the method of feature_extraction
    :param gt_labels: corresponding ground truth labels
    """
    vocab_path = os.path.join(feature_save_dir, 'drebin.vocab')
    vocab_info_path = os.path.join(feature_save_dir, 'drebin.vocab_info')
    if overwrite or (not os.path.exists(vocab_path)):
        assert len(feature_path_list) == len(gt_labels)
        features = load_features(feature_path_list)
        tmp_vocab, tmp_vocab_info, features_cln = get_vocabulary(features)
        # we select 10,000 features
        vocab, sel_vocab_info = feature_selection(features, gt_labels, tmp_vocab, tmp_vocab_info, dim=10000)
        utils.dump_pickle(vocab, vocab_path)
        utils.dump_pickle(sel_vocab_info, vocab_info_path)
    else:
        vocab = utils.read_pickle(vocab_path)
        features = load_features(feature_path_list)
        features_cln = get_feature_simf(features)

    dataX_np = get_feature_representation(features_cln, vocab)
    return dataX_np, gt_labels


def get_word_category(vocabulary, vocabulary_info, defined_comp):
    """
    Get the category for each word in vocabulary, based on the COMP in conf file
    :rtype: object
    """

    def _api_check(dalvik_code_line_list):
        for code_line in dalvik_code_line_list:
            invoke_match = re.search(
                r'(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeParam>([vp0-9,. ]*?)), (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
                code_line)
            if invoke_match is None:
                return defined_comp['Notdefined']
            if invoke_match.group('invokeType') == 'invoke-virtual' or invoke_match.group(
                    'invokeType') == 'invoke-virtual/range' or \
                    invoke_match.group('invokeType') == 'invoke-static' or \
                    invoke_match.group('invokeType') == 'invoke-static/range':
                if invoke_match.group('invokeObject').startswith('Landroid'):
                    return defined_comp['Android_API']
                elif invoke_match.group('invokeObject').startswith('Ljava'):
                    return defined_comp['Java_API']
                else:
                    return defined_comp['Notdefined']
            else:
                return defined_comp['Notdefined']

    word_cat_dict = collections.defaultdict()
    for w in vocabulary:
        if 'activity_list_' in w:
            word_cat_dict[w] = defined_comp['Activity']
        elif 'requested_permission_list_' in w:
            word_cat_dict[w] = defined_comp['Permission']
        elif 'service_list_' in w:
            word_cat_dict[w] = defined_comp['Service']
        elif 'content_provider_list_' in w:
            word_cat_dict[w] = defined_comp['Provider']
        elif 'broadcast_receiver_list_' in w:
            word_cat_dict[w] = defined_comp['Receiver']
        elif 'hardware_list_' in w:
            word_cat_dict[w] = defined_comp['Hardware']
        elif 'intentfilter_list_' in w:
            word_cat_dict[w] = defined_comp['Intentfilter']
        elif 'used_permission_list_' in w:
            word_cat_dict[w] = defined_comp['Notdefined']
        elif 'restricted_api_list_' in w:
            word_cat_dict[w] = _api_check(vocabulary_info[w])
        elif 'suspicious_api_list' in w:
            word_cat_dict[w] = _api_check(vocabulary_info[w])
        elif 'url_list' in w:
            word_cat_dict[w] = defined_comp['User_String']
        else:
            word_cat_dict[w] = defined_comp['Notdefined']
    return word_cat_dict


def get_vocab_constrains(feature_save_dir):
    vocab_path = os.path.join(feature_save_dir, 'drebin.vocab')
    vocab = utils.read_pickle(vocab_path)

    vocab_info_path = os.path.join(feature_save_dir, 'drebin.vocab_info')
    vocab_info = utils.read_pickle(vocab_info_path)

    word_dict = get_word_category(vocab, vocab_info, COMP)

    insertion_array = np.zeros(len(vocab), )
    removal_array = np.zeros(len(vocab), )

    for i, word in enumerate(vocab):
        cat = word_dict.get(word)
        if cat is not None:
            if cat in INSTR_ALLOWED[OPERATOR[0]]:
                insertion_array[i] = 1
            else:
                insertion_array[i] = 0
            if cat in INSTR_ALLOWED[OPERATOR[1]]:
                removal_array[i] = 1
            else:
                removal_array[i] = 0
        else:
            raise ValueError("Incompatible value.")

    return insertion_array, removal_array


def _main():
    args = cmd_md.parse_args()
    malware_dir = os.path.join(args.dataset_dir, 'malicious_samples')
    malware_feature_files = get_drebin_features(malware_dir, args.feature_dir, args.overwrite)

    benware_dir = os.path.join(args.dataset_dir, 'benign_samples')
    benware_feature_files = get_drebin_features(benware_dir, args.feature_dir, args.overwrite)

    # get representations
    feature_files = malware_feature_files + benware_feature_files
    gt_labels = np.zeros((len(feature_files),), dtype=np.int32)
    gt_labels[:len(malware_feature_files)] = 1

    # get train-val-test
    def _get_base_name(paths):
        return [os.path.basename(path_) for path_ in paths]

    data_split_path = os.path.join(args.dataset_dir, "tr_te_va.split")
    if os.path.exists(data_split_path):
        (train_dn, train_y), (validation_dn, validation_y), (test_dn, test_y) = utils.read_pickle(data_split_path)
    else:
        train_x, test_x, train_y, test_y = train_test_split(feature_files, gt_labels, test_size=0.2,
                                                            random_state=23456, shuffle=True)
        train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y,
                                                                        test_size=0.25,
                                                                        random_state=23456, shuffle=True)
        # get names
        train_dn, validation_dn, test_dn = _get_base_name(train_x), _get_base_name(validation_x), _get_base_name(
            test_x)
        utils.dump_pickle(((train_dn, train_y), (validation_dn, validation_y), (test_dn, test_y)), data_split_path)

    train_x = [os.path.join(args.feature_dir, train_name) for train_name in train_dn]
    validation_x = [os.path.join(args.feature_dir, val_name) for val_name in validation_dn]
    test_x = [os.path.join(args.feature_dir, test_name) for test_name in test_dn]

    feature_save_dir = os.path.join(args.dataset_dir, 'drebin')
    utils.mkdir(feature_save_dir)
    train_np_x, train_y = feature_preprocess(train_x, train_y, feature_save_dir, args.overwrite)

    insertion, removal = get_vocab_constrains(feature_save_dir)

    validation_np_x, validation_y = feature_preprocess(validation_x, validation_y, feature_save_dir, args.overwrite)
    test_np_x, test_y = feature_preprocess(test_x, test_y, feature_save_dir, args.overwrite)

    utils.dump_pickle((train_np_x, train_y), path=os.path.join(feature_save_dir, 'train.pkl'))
    utils.dump_pickle((validation_np_x, validation_y), path=os.path.join(feature_save_dir, 'validation.pkl'))
    utils.dump_pickle((test_np_x, test_y), path=os.path.join(feature_save_dir, 'test.pkl'))
    np.savez(os.path.join(feature_save_dir, 'constraints.npz'), insertion=insertion, removal=removal)

    return 0


if __name__ == "__main__":
    _main()
