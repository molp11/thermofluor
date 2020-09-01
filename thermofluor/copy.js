//copy to Clipboard
function fallbackCopyTextToClipboard(text) {
  var textArea = document.createElement("textarea");
  textArea.value = text;
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    var successful = document.execCommand('copy');
    var msg = successful ? 'successful' : 'unsuccessful';
    console.log('Fallback: Copying text command was ' + msg);
  } catch (err) {
    console.error('Fallback: Oops, unable to copy', err);
  }

  document.body.removeChild(textArea);
}
function copyTextToClipboard(text) {
  if (!navigator.clipboard) {
    fallbackCopyTextToClipboard(text);
    return;
  }
  navigator.clipboard.writeText(text).then(function() {
    console.log('Async: Copying to clipboard was successful!');
  }, function(err) {
    console.error('Async: Could not copy text: ', err);
  });
}

var data = source.data;
var copytext = '';
var state = state;
for (var i = 0; i < data['Sample'].length; i++) {
    if (state == 0) {
        var tmpx = data['Sample'][i].concat(' ');
        var tmpy = data['Tm'][i].toString();
        var tmprow = tmpx.concat(tmpy);
        var currRow = [tmprow.concat('\n')];
    }
    if (state == 1) {
        var tmpx = data['Sample'][i].concat(' ');
        var tmpy = data['Tm-1'][i].toString().concat(' ');
        var tmpz = data['Tm-2'][i].toString();
        var tmprow = tmpx.concat(tmpy);
        tmprow = tmprow.concat(tmpz);
        var currRow = [tmprow.concat('\n')];
    }
    if (state == 2) {
        var tmpx = data['Sample'][i].concat(' ');
        var tmpy = data['Tm'][i].toString().concat(' ');
        var tmpz = data['Tm-1'][i].toString().concat(' ');
        var tmpw = data['Tm-2'][i].toString();
        var tmprow = tmpx.concat(tmpy);
        tmprow = tmprow.concat(tmpz);
        tmprow = tmprow.concat(tmpw);
        var currRow = [tmprow.concat('\n')];
    }
    var joined = currRow.join();
    copytext = copytext.concat(joined);
}

copyTextToClipboard(copytext)
