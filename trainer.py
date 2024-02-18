


<!DOCTYPE HTML>
<html>

<head>
    <meta charset="utf-8">

    <title>JupyterHub</title>
    <meta http-equiv="X-UA-Compatible" content="chrome=1">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    
    <link rel="stylesheet" href="/hub/static/css/style.min.css?v=bff49b4a161afb17ee3b71927ce7d6c4e5b0e4b9ef6f18ca3e356a05f29e69776d3a76aee167060dd2ae2ee62d3cfdcf203b4b0090b1423f7d629ea7daa3f9da" type="text/css"/>
    
    
    <link rel="icon" href="/hub/static/favicon.ico?v=fde5757cd3892b979919d3b1faa88a410f28829feb5ba22b6cf069f2c6c98675fceef90f932e49b510e74d65c681d5846b943e7f7cc1b41867422f0481085c1f" type="image/x-icon">
    
    
    <script src="/hub/static/components/requirejs/require.js?v=bd1aa102bdb0b27fbf712b32cfcd29b016c272acf3d864ee8469376eaddd032cadcf827ff17c05a8c8e20061418fe58cf79947049f5c0dff3b4f73fcc8cad8ec" type="text/javascript" charset="utf-8"></script>
    <script src="/hub/static/components/jquery/dist/jquery.min.js?v=f3de1813a4160f9239f4781938645e1589b876759cd50b7936dbd849a35c38ffaed53f6a61dbdd8a1cf43cf4a28aa9fffbfddeec9a3811a1bb4ee6df58652b31" type="text/javascript" charset="utf-8"></script>
    <script src="/hub/static/components/bootstrap/dist/js/bootstrap.min.js?v=a014e9acc78d10a0a7a9fbaa29deac6ef17398542d9574b77b40bf446155d210fa43384757e3837da41b025998ebfab4b9b6f094033f9c226392b800df068bce" type="text/javascript" charset="utf-8"></script>
    
    <script>
      require.config({
          
          urlArgs: "v=20231109221008",
          
          baseUrl: '/hub/static/js',
          paths: {
            components: '../components',
            jquery: '../components/jquery/dist/jquery.min',
            bootstrap: '../components/bootstrap/dist/js/bootstrap.min',
            moment: "../components/moment/moment",
          },
          shim: {
            bootstrap: {
              deps: ["jquery"],
              exports: "bootstrap"
            },
          }
      });
    </script>

    <script type="text/javascript">
      window.jhdata = {
        base_url: "/hub/",
        prefix: "/",
        
        
        admin_access: false,
        
        
        options_form: false,
        
      }
    </script>

    
    

</head>

<body>

<noscript>
  <div id='noscript'>
    JupyterHub requires JavaScript.<br>
    Please enable it to proceed.
  </div>
</noscript>


  <nav class="navbar navbar-default">
    <div class="container-fluid">
      <div class="navbar-header">
        
        <span id="jupyterhub-logo" class="pull-left">
            <a href="/hub/"><img src='/hub/logo' alt='JupyterHub' class='jpy-logo' title='Home'/></a>
        </span>
        
        
      </div>

      <div class="collapse navbar-collapse" id="thenavbar">
        
        <ul class="nav navbar-nav navbar-right">
          
            <li>
              

            </li>
          
        </ul>
      </div>

      
      
    </div>
  </nav>











<div id="login-main" class="container">


<form action="/hub/login?next=%2Fhub%2Fapi%2Foauth2%2Fauthorize%3Fclient_id%3Djupyterhub-user-sbody1%26redirect_uri%3D%252Fuser%252Fsbody1%252Foauth_callback%26response_type%3Dcode%26state%3DeyJ1dWlkIjogImVlNmQ1ZjljN2JlNzQwMTBhMDM4MWI5OTIxZTFjNTY4IiwgIm5leHRfdXJsIjogIi91c2VyL3Nib2R5MS9maWxlcy9ZSC9Wb2xhdGlsaXR5L1ZvbGF0aWxpdHlfMjAyMy9jb21tb24vdHJhaW5lci5weT9kb3dubG9hZD0xIn0" method="post" role="form">
  <div class="auth-form-header">
    Sign in
  </div>
  <div class='auth-form-body'>

    <p id='insecure-login-warning' class='hidden'>
    Warning: JupyterHub seems to be served over an unsecured HTTP connection.
    We strongly recommend enabling HTTPS for JupyterHub.
    </p>

    
    <label for="username_input">Username:</label>
    <input
      id="username_input"
      type="text"
      autocapitalize="off"
      autocorrect="off"
      class="form-control"
      name="username"
      val=""
      tabindex="1"
      autofocus="autofocus"
    />
    <label for='password_input'>Password:</label>
    <input
      type="password"
      class="form-control"
      name="password"
      id="password_input"
      tabindex="2"
    />

    <div class="feedback-container">
      <input
        id="login_submit"
        type="submit"
        class='btn btn-jupyter'
        value='Sign in'
        tabindex="3"
        />
      <div class="feedback-widget hidden">
        <i class="fa fa-spinner"></i>
      </div>
    </div>

    
    
    

  </div>
</form>


</div>










<div class="modal fade" id="error-dialog" tabindex="-1" role="dialog" aria-labelledby="error-label" aria-hidden="true">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal"><span aria-hidden="true">&times;</span><span class="sr-only">Close</span></button>
        <h4 class="modal-title" id="error-label">Error</h4>
      </div>
      <div class="modal-body">
        
  <div class="ajax-error">
    The error
  </div>

      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-default" data-dismiss="modal">Cancel</button>
        <button type="button" class="btn btn-primary" data-dismiss="modal" data-dismiss="modal">OK</button>
      </div>
    </div>
  </div>
</div>





<script>
if (window.location.protocol === "http:") {
  // unhide http warning
  var warning = document.getElementById('insecure-login-warning');
  warning.className = warning.className.replace(/\bhidden\b/, '');
}
// setup onSubmit feedback
$('form').submit((e) => {
  var form = $(e.target);
  form.find('.feedback-container>input').attr('disabled', true);
  form.find('.feedback-container>*').toggleClass('hidden');
  form.find('.feedback-widget>*').toggleClass('fa-pulse');
});
</script>


</body>

</html>