import sys
import xml.etree.ElementTree as ET

# other possible tracks
#     {'name': 'dirt-1', 'category': 'dirt'},
#     {'name': 'dirt-2', 'category': 'dirt'},
#     {'name': 'dirt-4', 'category': 'dirt'},
#     {'name': 'dirt-6', 'category': 'dirt'},
#     {'name': 'mixed-2', 'category': 'dirt'},


def main():
    tracks = {
        'easy': [
            {'name': 'forza', 'category': 'road'},
            {'name': 'e-track-4', 'category': 'road'},
            {'name': 'g-track-1', 'category': 'road'},
            {'name': 'c-speedway', 'category': 'oval'},
        ],
        'medium': [
            {'name': 'ruudskogen', 'category': 'road'},
            {'name': 'aalborg', 'category': 'road'},
            {'name': 'e-track-2', 'category': 'road'},
            {'name': 'a-speedway', 'category': 'oval'},
        ],
        'hard': [
            {'name': 'alpine-1', 'category': 'road'},
            {'name': 'wheel-2', 'category': 'road'},
            {'name': 'e-track-2', 'category': 'road'},
            {'name': 'e-track-5', 'category': 'oval'},
        ],
        'validation': [
            {'name': 'g-track-2', 'category': 'road'},
            {'name': 'alpine-2', 'category': 'road'},
            {'name': 'michigan', 'category': 'oval'},
            {'name': 'b-speedway', 'category': 'oval'},
        ]
    }

    track = tracks[sys.argv[1]][int(sys.argv[2])]
    xml = ET.parse('./simulator/quickrace.xml')
    root = xml.getroot()
    print('current track: {}-{}'.format(track['category'], track['name']))
    for node in root.findall('section'):
        if node.attrib['name'] == 'Tracks':
            for attstr in node.find('section'):
                if attstr.attrib['name'] == 'name':
                    attstr.set('val', track['name'])
                elif attstr.attrib['name'] == 'category':
                    attstr.set('val', track['category'])
            break
    return xml.write('./simulator/quickrace.xml', encoding='UTF-8')


if __name__ == '__main__':
    main()

