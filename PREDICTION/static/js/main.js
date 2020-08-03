$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('.result').hide();
    $('#output-image').hide();
    

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            console.log('i m here');
            console.log('i m here', typeof(input.files[0]));
            console.log(input.files[0]['name']);
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $("#imagePreview").attr("data-item",input.files[0]['name']);

                $('#imagePreview').hide();
                
                
                $('#imagePreview').fadeIn(650);
                $('#output-image').hide();
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('.result').text('');
        $('.key').text('');
        $('.result').hide();
        $("#output-image").hide();
        
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        console.log('form_data',form_data);
        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                /*var imageUrl = "C:/Users/PRINCE/Documents/PREDICTION/static/1.jpg";
                $("#output-image").css("background-image", "url(" + imageUrl + ")");
                */
                console.log('data',data.split('//#***#//'));
                var result_data = data.split('//#***#//');
                $('.loader').hide();
                $('.result').fadeIn(600);
                $('.result').text(result_data[0]);
                $('.key').text(result_data[1]);
                console.log('abstract',data);
                
                $('.container1').show();
                
                $('.result').show();
                
                $('#output-image').show();
                console.log('value',$("#imagePreview").attr("data-item"
                ));
                

                $("#output-image").css("background-image", "");
                var image_path = "static/images/"+$("#imagePreview").attr("data-item");
                console.log('image-url',image_path);
                $("#output-image").css("background-image", "url(" + image_path + ")");
                console.log('Success!');
            },
        });
    });

});
