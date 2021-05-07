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
var copytext = 'Well ID,Sample name,Tm,Error\n';
for (var i = 0; i < data['w'].length; i++) {
    var currRow = [data['w'][i],
                   data['n'][i],
                   data['t'][i],
                   data['e'][i].toString().concat('\n')];
    var joined = currRow.join();
    copytext = copytext.concat(joined);
}

copyTextToClipboard(copytext)
