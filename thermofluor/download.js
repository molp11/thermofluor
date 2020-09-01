var data = source.data;
var state = state;
var filetext = '';
if (state == 0) {
  filetext = 'Sample,Tm\n';
}
if (state == 1) {
  filetext = 'Sample,Tm 1,Tm 2\n';
}
if (state == 2) {
  filetext = 'Sample,Tm,AIC monophasic,AIC biphasic,Max probability monophasic\n';
}
for (var i = 0; i < data['Sample'].length; i++) {
    if (state == 0) {
        var currRow = [data['Sample'][i],
                       data['Tm'][i].toString().concat('\n')];
    }
    if (state == 1) {
        var currRow = [data['Sample'][i],
                       data['Tm-1'][i].toString(),
                       data['Tm-2'][i].toString().concat('\n')];
    }
    if (state == 2) {
        var currRow = [data['Sample'][i],
                       data['Tm'][i].toString(),
                       data['Tm-1'][i].toString(),
                       data['Tm-2'][i].toString(),
                       data['AIC monophasic'][i].toString(),
                       data['AIC biphasic'][i].toString(),
                       data['probability'][i].toString().concat('\n')];
    }
    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = file_name;
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
} else {
    var link = document.createElement("a");
    link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'));
}
