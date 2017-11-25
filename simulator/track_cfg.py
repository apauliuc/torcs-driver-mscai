import sys
import xml.etree.ElementTree as ET

def main():
    tracks = [
        {},
        {'name': 'forza', 'category': 'road'},
        {'name': 'g-track-1', 'category': 'road'},
        {'name': 'g-track-2', 'category': 'road'},
        {'name': 'ole-road-1', 'category': 'road'},
        {'name': 'ruudskogen', 'category': 'road'},
        {'name': 'spring', 'category': 'road'},
        {'name': 'wheel-1', 'category': 'road'},
        {'name': 'aalborg', 'category': 'road'},
        {'name': 'apline-1', 'category': 'road'},
        {'name': 'e-track-2', 'category': 'road'},
        {'name': 'dirt-1', 'category': 'dirt'},
        {'name': 'dirt-2', 'category': 'dirt'},
        {'name': 'dirt-4', 'category': 'dirt'},
        {'name': 'dirt-6', 'category': 'dirt'},
        {'name': 'mixed-2', 'category': 'dirt'},
        {'name': 'a-speedway', 'category': 'oval'},
        {'name': 'd-speedway', 'category': 'oval'},
        {'name': 'e-track-5', 'category': 'oval'},
        {'name': 'michigan', 'category': 'oval'},
        {'name': 'b-speedway', 'category': 'oval'},

    ]
    track = tracks[int(sys.argv[1])]

    xml = ET.parse('quickrace.xml')
    root = xml.getroot()
    print('Changing track to category={} name={}'.format(track['category'], track['name']))
    for node in root.findall('section'):
        if node.attrib['name'] == 'Tracks':
            for attstr in node.find('section'):
                if attstr.attrib['name'] == 'name':
                    attstr.set('val', track['name'])
                elif attstr.attrib['name'] == 'category':
                    attstr.set('val', track['category'])
            break


if __name__ == '__main__':
    main()

