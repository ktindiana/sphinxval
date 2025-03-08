import os

from . import config

def add_banner(banner_text):
    html = ''
    html += '    <div class="banner">\n'
    html += '        ' + banner_text + '\n'
    html += '    </div>\n'
    return html

def add_banner_style():
    html = '' 
    html += '        .banner {\n'
    html += '            background-color: #E6982B;\n'
    html += '            color: black;\n'
    html += '            padding: 15px 20px;\n'
    html += '            text-align: center;\n'
    html += '            font-size: 24px;\n'
    html += '            width: 100%;\n'
    html += '            box-sizing: border-box;\n'
    html += '        }\n'
    return html

def add_body_style():
    html = ''
    html += '        body {\n'
    html += '            font-family: Arial, sans-serif;\n'
    html += '            margin: 0;\n'
    html += '            padding: 0;\n'
    html += '            background-color: #f4f4f4;\n'
    html += '        }\n'
    return html

def add_header_style():
    html = ''
    html += '        .header {\n'
    html += '            background-color: #E6982B;\n'
    html += '            color: white;\n'
    html += '            padding: 10px 0;\n'
    html += '            text-align: center;\n'
    html += '        }\n'
    return html

def add_content_style():
    html = ''
    html += '        .content {\n'
    html += '            padding: 20px;\n'
    html += '        }\n'
    return html

def add_link_style():
    html = ''
    html += '        .link a {\n'
    html += '            display: block;\n'
    html += '            margin: 10px 0;\n'
    html += '            color: #0000ff;\n'
    html += '            text-decoration: none;\n'
    html += '        }\n'
    html += '        .link a:hover{\n'
    html += '            text-decoration: underline;\n'
    html += '        }\n'
    return html

def add_style():
    html = ''
    html += add_body_style()
    html += add_header_style()
    html += add_content_style()
    html += add_banner_style()
    html += add_link_style()
    return html

def make_index(directory, title='SPHINX Validation Report Repository', banner_text=None):
    html = ''
    html += '<!DOCTYPE HTML>\n'
    html += '<html lang="en">\n'
    html += '<head>\n'
    html += '    <meta charset="UTF-8">\n'
    html += '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    html += '    <title>' + title + '</title>\n'
    html += '    <style>\n'
    html += add_style()
    html += '    </style>\n'
    html += '</head>\n'
    html += '<body>\n'
    if banner_text is not None:
        html += add_banner(banner_text)
    
    # GET FILES AND DIRECTORIES
    files = os.listdir(directory)
    # REMOVE MARKDOWN FILES
    files = [file for file in files if not file.endswith('.md')]
    files = [file for file in files if not file.endswith('index.html')]
    files.sort()
    html += '    <div class="content">\n'
    html += '        <div class="links">\n'
    for file in files:
        if config.baseurlpath is not None:
            file = os.path.normpath(os.path.join(config.baseurlpath, file))
        html += '            <a href="' + file + '">' + os.path.basename(file) + '</a><br>\n'
    html += '        </div>\n'
    html += '    </div>\n'
    html += '</body>\n'
    html += '</html>'
    return html
